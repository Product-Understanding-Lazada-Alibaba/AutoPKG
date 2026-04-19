import os
import re
import copy
import asyncio
import requests
import uuid
import logging
import time
import functools
from collections import defaultdict

import torch
import numpy as np
import pandas as pd
from openlm_hub import repo_download

from .odps_utils import load_table, save_table, drop_partition
from .simple_kg import SimpleKG
from .kgd_prompts import KGD_AM_PROMPT, TYPE_DESC_PROMPT, KEY_DESC_PROMPT, KGD_AMRD_PROMPT
from .data import NodeData, EdgeData, LogEntry, UpdateRequest
from .engines import EmbeddingEngine, AgentEngine
from .debug_utils import _print_debug_prompt_action

MASTER_ADDR = os.environ.get("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = int(os.environ.get("MASTER_PORT", 8000))

logger = logging.getLogger(__name__)

def timed_async(label: str | None = None):
    def deco(fn):
        name = label or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return await fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                logger.info(f"{name} completed in {elapsed:.3f}s")
        return wrapper
    return deco

class KGD:
    def __init__(
        self, 
        task=None,
        rank=0,
        world_size=1,
        emb_model_name=None, 
        agent_model_name=None, 
        node_input=None, 
        node_output=None, 
        edge_input=None, 
        edge_output=None, 
        log_table=None, 
        debug_rows=0,
    ):
        self.task = task
        self.rank = rank
        self.world_size = world_size
        self.node_input = node_input
        self.node_output = node_output
        self.edge_input = edge_input
        self.edge_output = edge_output
        self.log_table = log_table
        self.debug_rows = debug_rows
        self.logs = []
        
        self.simple_kg = SimpleKG(embedding_dim=1024)
        
        # Async Queues and Locks
        self.input_queue = asyncio.Queue(maxsize=128)
        self.action_queue = asyncio.Queue(maxsize=1024)
        self.type_locks = defaultdict(asyncio.Lock)
        self.debug_lock = asyncio.Lock()
        self.kg_lock = asyncio.Lock()
        self.background_tasks = []

        assert self.task in ['type', 'key', 'value']

        if not torch.cuda.is_available():
            print("No GPU available")
            return

        # --- Model Download ---
        emb_model_path = repo_download(
            emb_model_name,
            ignore_patterns=["global_step*/*_states.pt", "*iter_0000001/dist_optimizer/*"],
            repo_meta={"usage": "model_name_or_path"},
            use_subprocess=True,
        )
        agent_model_path = repo_download(
            agent_model_name,
            ignore_patterns=["global_step*/*_states.pt", "*iter_0000001/dist_optimizer/*"],
            repo_meta={"usage": "model_name_or_path"},
            use_subprocess=True,
        )

        # --- Async Engine Setup ---
        
        self._emb_engine = EmbeddingEngine(emb_model_path)
        self._agent_engine = AgentEngine(agent_model_path)

    # --- Async Infrastructure ---

    async def start_processing(self):
        """Starts the background worker and updater loops."""
        # Create multiple workers for parallel processing
        # Note: Workers will be throttled by node_type locks
        for _ in range(128):
            self.background_tasks.append(asyncio.create_task(self._worker_loop()))
        
        # Single updater to maintain KG consistency
        self.background_tasks.append(asyncio.create_task(self._updater_loop()))
        
        # Open route on master node for distributed inference
        self._register_route()

    async def stop_processing(self):
        """Waits for queues to empty and cancels background tasks."""
        await self.input_queue.join()
        await self.action_queue.join()
        
        for task in self.background_tasks:
            task.cancel()

    async def _worker_loop(self):
        """
        Worker loop:
        - Pulls an item from input_queue
        - Uses locks to prevent concurrent processing of the same key
        - Runs the main LLM/pipeline logic
        - Enqueues the result for the updater and waits until the updater commits it
        """
        while True:
            try:
                item = await self.input_queue.get()
                node_type = item['node_type']
                node_name = item['node_name']

                if not self.is_selected(node_type):
                    continue

                # Lock (fine-grained): prevent two workers from processing the same
                # (node_type, node_name) at the same time.
                async with self.type_locks[(node_type, node_name)]:
                    
                    # Step 1: main processing (e.g., exact match -> update properties -> embed)
                    result = await self._process_logic(item)

                # Lock (coarse-grained): serialize any downstream decision-making / updates
                # for this node_type so that candidate checks + update scheduling happen
                # in a consistent order.
                async with self.type_locks[node_type]:

                    if not result.node_id:
                        # If no node_id was produced, run candidate checking to decide
                        # whether we should create a new node or attach to an existing one.
                        r = await self._check_candidate(result.new_node, result.embedding)
                        
                        result.log_entry.prompt = r['prompt']
                        result.log_entry.action = r['action']
                        result.log_entry.thought = r['thought']
                        
                        action_split = result.log_entry.action.split()
                        result.action = action_split[0]
                        if len(action_split) > 1:
                            result.node_id = action_split[1]

                    # Step 2: hand off to the updater. We block (while holding the lock)
                    # until the updater signals it has finished persisting the result.
                    update_done_event = asyncio.Event()
                    
                    await self.action_queue.put({
                        'result': result,
                        'done_event': update_done_event
                    })
                    
                    # Wait until the updater commits/writes this result before allowing
                    # another worker to proceed under this lock.
                    await update_done_event.wait()
                
                # Mark this input item as fully processed.
                self.input_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in worker loop")

                # If we already created the event, ensure it is released so the updater/
                # waiters don't hang forever on errors.
                if 'update_done_event' in locals():
                    update_done_event.set()
                
                # Ensure the queue item is acknowledged even on failure.
                self.input_queue.task_done()

    async def _updater_loop(self):
        """
        Sequential loop to apply changes to SimpleKG.
        """
        while True:
            try:
                action_item = await self.action_queue.get()
                result = action_item['result']
                done_event = action_item['done_event']

                # Apply changes to KG (Sequential)
                await self._apply_action(result)

                # Signal worker to release the type lock
                done_event.set()
                self.action_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in updater loop")
                if 'done_event' in locals():
                    done_event.set()

    # --- Distributed Inference ---

    def _register_route(self):
        if self.rank == 0 and self.world_size > 1 and self.task == 'value':
            
            from fastapi import FastAPI
            import uvicorn
            import threading

            app = FastAPI()
            main_loop = asyncio.get_event_loop() 

            @app.post("/add")
            async def add(node_data: NodeData):

                async def safe_add():
                    # Now we are running in the Main Loop context
                    # We can safely use asyncio.Lock here if it was created in Main Loop
                    async with self.kg_lock: 
                        node_id = self.simple_kg.next_node_id(self.task)
                        node_data.node_id = node_id
                        self.simple_kg.add_node(node_data)
                    return {"node_id": node_id}

                future = asyncio.run_coroutine_threadsafe(safe_add(), main_loop)
                return future.result()

            def run_server():
                try:
                    uvicorn.run(
                        app, 
                        host="0.0.0.0", 
                        port=MASTER_PORT, 
                        log_level="warning",
                    )

                except Exception as e:
                    print(f"[Master] Server error: {e}")
                    # Optional: Propagate error or set a flag to stop the main process
                    import sys
                    sys.exit(1)
            
            server_thread = threading.Thread(target=run_server, daemon=True)
            server_thread.start()
            logger.info(f"[Master] HTTP Server started in thread on port {MASTER_PORT}")
            
    def _update_master(self, node_data: NodeData):
        if self.rank != 0 and self.world_size > 1 and self.task == 'value':

            url = f"http://{MASTER_ADDR}:{MASTER_PORT}/add"
            max_retries = 10
            retry_delay = 5

            # Convert Pydantic model to a JSON-compatible dict
            # mode='json' ensures enums, datetimes, etc., are converted to primitive types
            payload = node_data.model_dump(mode='json')

            for i in range(max_retries):
                try:
                    response = requests.post(url, json=payload, timeout=30)
                    if response.status_code == 200:
                        return response.json()
                except requests.exceptions.RequestException as e:
                    print(f"[Worker] Connection failed (Attempt {i+1}/{max_retries}): {e}")

                if i < max_retries - 1:
                    time.sleep(retry_delay)

    # --- Data Loading / Saving ---

    def is_selected(self, s: str) -> int:
        """ running values on KGD is very slow, so I divide them by type and run in parallel """
        if self.world_size == 1:
            return True
        return abs(hash(s)) % self.world_size == self.rank
    
    def pull_graph(self):
        self._load_nodes()

    def push_graph(self):
        if self.rank == 0:
            self._save_nodes()
        self._save_edges()
        self._save_logs()

    @timed_async()
    async def embed_nodes(self):
        nodes = [d for n, d in self.simple_kg.get_nodes(task=self.task)]
        nodes = [d for d in nodes if self.is_selected(d.node_type)]
        tasks = [self._emb_engine.embed(d.node_name) for d in nodes]
        embeddings = await asyncio.gather(*tasks)
        self.simple_kg.update_embeddings_batch(nodes, embeddings)
    
    @timed_async()
    async def update_nodes(self):
        nodes = self.simple_kg.get_nodes(task=self.task)
        tasks = [self._update_properties(d) for n, d in nodes if '_needs_update' in d.properties]
        await asyncio.gather(*tasks)

    # --- Core Logic ---

    async def __call__(self, node_name, node_type, properties, edges):
        """
        Adds the input to the queue. Returns immediately.
        """
        await self.input_queue.put({
            'node_name': node_name,
            'node_type': node_type,
            'properties': properties,
            'edges': edges
        })

    async def _process_logic(self, item):
        """
        The logic that runs inside the worker (inside the lock).
        Returns a dict of data needed for the Update step.
        """
        node_name = item['node_name']
        node_type = item['node_type']
        properties = item['properties']
        
        new_node = NodeData(
            node_type=node_type,
            node_name=node_name, 
            properties=properties, 
            task=self.task
        )
        
        log_entry = LogEntry(node_type=node_type, node_name=node_name)
        
        # 1. Quick check for exact match (Read Only)
        node_id = self.simple_kg.get_node_id(node_name=node_name, node_type=node_type)
        embedding = None
        action = None

        if node_id:
            new_node.node_id = node_id
            log_entry.node_id = node_id
            log_entry.action = f"MERGE {node_id}"
            action = "MERGE"

            # Flag: update properties
            async with self.type_locks[node_id]:
                node_data = self.simple_kg.get_node_data(node_id)
                if '_needs_update' in node_data.properties:
                    await self._update_properties(new_node)
                    node_data.properties.pop('_needs_update')
        else:
            # Update properties using Agent
            await self._update_properties(new_node)

            # Embedding
            embedding = await self._emb_engine.embed(node_name)

        return UpdateRequest(
            action=action,
            node_id=node_id,
            new_node=new_node,
            embedding=embedding,
            log_entry=log_entry,
            edges=item['edges']
        )

    async def _check_candidate(self, node_data: NodeData, embedding):
        relevant_nodes = self.simple_kg.get_most_similar_nodes(
            embedding, 
            node_data.node_type, 
            topk=10
        )
        pretty_nodes = [n.to_candidate() for n in relevant_nodes]
        pretty_candidate = node_data.to_candidate()
        # prompt = KGD_AMRD_PROMPT.format(
        prompt = KGD_AM_PROMPT.format(
            pretty_nodes=pretty_nodes, 
            node_type=node_data.node_type, 
            pretty_candidate=pretty_candidate
        )

        choices = ["ADD"] + [f"MERGE {n.node_id}" for n in relevant_nodes]
        result = await self._agent_engine.generate(prompt, choices=choices)
        thought, action = self._split_thought_action(result['output'])

        async with self.debug_lock:
            if self.debug_rows > 0:
                self.debug_rows -= 1
                _print_debug_prompt_action(result['input'], action)

        return {
            'prompt': prompt,
            'action': action,
            'thought': thought,
        }

    async def _update_properties(self, node_data):
        def _extract(pattern, x):
            match = re.search(pattern, x, re.DOTALL)
            return match.group(1).strip() if match else None
        
        if 'description' in node_data.properties:
            return
        
        if self.task == 'type':
            prompt = TYPE_DESC_PROMPT.format(product_type=node_data.node_name)
            result = await self._agent_engine.generate(prompt)
            node_data.properties['description'] = result['output'].strip()
        
        elif self.task == 'key':
            prompt = KEY_DESC_PROMPT.format(attribute_key=node_data.node_name)
            result = await self._agent_engine.generate(prompt)
            node_data.properties['description'] = _extract(r'Description: ([^\n]+)', result['output'])
            node_data.properties['examples'] = _extract(r'Examples: ([^\n]+)', result['output'])

    async def _apply_action(self, result):
        """
        The sequential logic that modifies the KG.
        """
        action = result.action
        node_id = result.node_id
        new_node = result.new_node
        embedding = result.embedding
        log_entry = result.log_entry
        edges_input = result.edges

        if isinstance(action, str):
            action = action.strip().upper()

        if action not in ("ADD", "MERGE", "REPLACE", "DISCARD"):
            print(f"Action is invalid: {action}")
            pass

        elif action == "ADD":
            if not node_id:
                if self.rank == 0:
                    async with self.kg_lock:
                        node_id = self.simple_kg.next_node_id(self.task)
                else:
                    node_id = self._update_master(new_node)['node_id']
            
            async with self.kg_lock:
                new_node.node_id = node_id
                self.simple_kg.add_node(new_node, embedding)

        elif action == "MERGE":
            if node_id:
                self.simple_kg.merge_node(
                    node_id=node_id, 
                    new_node=new_node
                )

        elif action == "REPLACE":
            if node_id:
                self.simple_kg.replace_node_name(
                    node_id=node_id, 
                    new_name=new_node.node_name, 
                    new_properties=new_node.properties,
                    new_embedding=embedding
                )

        # Update Log
        log_entry.node_id = node_id
        self.logs.append(log_entry)

        # Add Relations
        if action in ['ADD', 'MERGE', 'REPLACE'] and node_id:
            for edge in edges_input:
                edge_data = EdgeData(**edge, task=self.task)
                if edge_data.source_node_id == 'this':
                    edge_data.source_node_id = node_id
                if edge_data.target_node_id == 'this':
                    edge_data.target_node_id = node_id

                # NOTE: Save generated data for frequency-based selection
                if 'description' in new_node.properties:
                    edge_data.properties['_gen_desc'] = new_node.properties['description']
                if 'examples' in new_node.properties:
                    edge_data.properties['_gen_examples'] = new_node.properties['examples']

                self.simple_kg.add_edge(edge_data)
                
    # --- Utilities ---

    def _split_thought_action(self, response: str):
        start_marker = "<think>"
        end_marker   = "</think>"

        start_idx = response.find(start_marker)
        end_idx   = response.find(end_marker)

        if start_idx == -1 and end_idx == -1:
            return None, response.strip()
        
        if start_idx == -1:
            thought = response[:end_idx].strip()
            action = response[end_idx + len(end_marker):].strip()
        else:
            thought = response[start_idx + len(start_marker):end_idx].strip()
            action = response[end_idx + len(end_marker):].strip()

        return thought, action
    
    def _load_nodes(self):
        df = load_table(self.node_input)
        if df.shape[0] > 0:
            for _, row in df.iterrows():
                self.simple_kg.add_node(NodeData.from_kwargs(**row))

    def _load_edges(self):
        df = load_table(self.edge_input)
        if df.shape[0] > 0:
            for _, row in df.iterrows():
                self.simple_kg.add_edge(EdgeData.from_kwargs(**row))

    def _save_nodes(self):
        nodes = self.simple_kg.get_nodes(task=self.task)
        rows = [node_data.to_dict(exclude_fields=['task']) for _, node_data in nodes]
        df = pd.DataFrame(rows)
        save_table(df, self.node_output, overwrite=True)

    def _save_edges(self):
        edges = self.simple_kg.get_edges(task=self.task)
        rows = [edge_data.to_dict(exclude_fields=['task']) for _, _, edge_data in edges]
        df = pd.DataFrame(rows)
        save_table(df, self.edge_output)

    def _save_logs(self):
        rows = [row.to_dict(exclude_fields=['task']) for row in self.logs]
        df = pd.DataFrame(rows)
        save_table(df, self.log_table)
        self.logs.clear()
