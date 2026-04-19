import faiss
import numpy as np
import networkx as nx
from datetime import datetime

from .data import NodeData, EdgeData

class IdMapper:
    def __init__(self):
        self.str2int = {}
        self.int2str = {}

    def get_int(self, sid: str) -> int:
        if sid not in self.str2int:
            new_id = len(self.str2int) + 1  # start at 1
            self.str2int[sid] = new_id
            self.int2str[new_id] = sid
        return self.str2int[sid]
    
    def get_str(self, iid: int) -> str:
        return self.int2str[iid] if iid in self.int2str else None

class SimpleKG:
    def __init__(self, embedding_dim):
        self.G = nx.DiGraph()
        self.embedding_dim = embedding_dim
                
        # Multiple FAISS indices by node_type using IndexIDMap
        self.indices = {}  # {node_type: faiss.IndexIDMap}
        self.id_mapper = IdMapper()
        
        self._max_id_counters = {}
        self._task_prefix = {'type': 'T', 'key': 'K', 'value': 'V'}
        self._idx = {}  # (node_type, node_name) -> node_id

    def _get_or_create_index(self, node_type):
        """Get or create FAISS IndexIDMap for node_type."""
        if node_type not in self.indices:
            base_index = faiss.IndexFlatL2(self.embedding_dim)
            index = faiss.IndexIDMap(base_index)
            self.indices[node_type] = index

        return self.indices[node_type]
    
    def _register_idx(self, node_data):
        if node_data.node_type and node_data.node_name and node_data.node_id:
            for name in [node_data.node_name] + node_data.synonyms:
                self._register_idx_synonym(
                    node_data.node_type, 
                    name, 
                    node_data.node_id
                )
    
    def _register_idx_synonym(self, node_type, node_name, node_id):
        self._idx[(node_type, node_name)] = node_id

    def next_node_id(self, task):
        prefix = self._task_prefix[task]

        if task not in self._max_id_counters:
            self._max_id_counters[task] = max(
                (int(n[len(prefix):]) for n, d in self.get_nodes(task=task) if n[len(prefix):].isdigit()),
                default=0,
            )
        
        next_num = self._max_id_counters[task] + 1
        return f"{prefix}{next_num:05d}"
    
    def add_node(
        self, 
        node_data: NodeData,
        embedding=None
    ):
        if self.has_node(node_data.node_id):
            return

        # Add node to NetworkX
        self.G.add_node(
            node_data.node_id, 
            data=node_data
        )
        self._register_idx(node_data)

        # Update counter
        task = node_data.task
        if task in self._max_id_counters:
            prefix = self._task_prefix[task]
            num = int(node_data.node_id[len(prefix):])
            if num > self._max_id_counters[task]:
                self._max_id_counters[task] = num

        if embedding is not None:
            self.update_embedding(node_data, embedding)

    def add_edge(
        self, 
        edge_data: EdgeData
    ):
        source_id = edge_data.source_node_id
        target_id = edge_data.target_node_id

        # Dummy nodes; won't be saved
        if not self.has_node(source_id):
            self.add_node(NodeData(node_id=source_id))
        if not self.has_node(target_id):
            self.add_node(NodeData(node_id=target_id))

        self.G.add_edge(
            source_id, 
            target_id, 
            data=edge_data
        )
    
    def update_embedding(self, node_data, embedding):
        embedding = np.array(embedding).astype(np.float32)
        assert embedding.shape == (self.embedding_dim,), f"Embedding must be of shape ({self.embedding_dim},)"

        # Add to FAISS
        id = self.id_mapper.get_int(node_data.node_id)
        index = self._get_or_create_index(node_data.node_type)
        index.remove_ids(np.array([id]))
        index.add_with_ids(np.expand_dims(embedding, axis=0), np.array([id]))

    def update_embeddings_batch(self, node_data_list, embedding_list):
        if len(node_data_list) != len(embedding_list):
            raise ValueError("node_data_list and embedding_list must have same length")
        
        # Convert embeddings to numpy array
        embeddings = np.array(embedding_list).astype(np.float32)
        if len(embedding_list) != 0:
            assert embeddings.shape == (len(node_data_list), self.embedding_dim)

        # Group by node_type since each type has its own FAISS index
        updates_by_type = {}
        for node_data, emb in zip(node_data_list, embeddings):
            node_type = node_data.node_type
            if node_type not in updates_by_type:
                updates_by_type[node_type] = {'node_datas': [], 'embeddings': []}
            updates_by_type[node_type]['node_datas'].append(node_data)
            updates_by_type[node_type]['embeddings'].append(emb)

        # Process each node_type separately
        for node_type, data in updates_by_type.items():
            index = self._get_or_create_index(node_type)
            node_datas = data['node_datas']
            embs = np.stack(data['embeddings'])  # Shape: (M, dim)

            # Map node_id → internal int ID
            faiss_ids = np.array([
                self.id_mapper.get_int(nd.node_id) for nd in node_datas
            ], dtype=np.int64)

            index.remove_ids(faiss_ids)
            index.add_with_ids(embs, faiss_ids)

    def merge_node(self, node_id, new_node):
        if not self.has_node(node_id):
            return
        node_data = self.get_node_data(node_id)
        node_data.add_synonym(new_node.node_name)
        self._register_idx_synonym(node_data.node_type, new_node.node_name, node_id)
        node_data.properties.update(new_node.properties)
    
    def replace_node_name(
        self, 
        node_id, 
        new_name, 
        new_properties,
        new_embedding
    ):
        if not self.G.has_node(node_id):
            return False

        # Update name and synonyms
        node_data = self.get_node_data(node_id)
        node_data.replace_node_name(new_name)
        node_data.update_properties(new_properties)
        self.update_embedding(node_data, new_embedding)
        self._register_idx(node_data)

    def get_most_similar_nodes(self, query_embedding, node_type=None, topk=20):
        """
        Search for most similar nodes.
        
        Args:
            query_embedding: np.array of shape (embedding_dim,) or (1, embedding_dim)
            node_type: Optional[str]; if provided, only search in this node_type's index
            topk: int; number of results to return
            
        Returns:
            List of dicts with keys: node_id, node_name, synonyms, distance
        """         
        query_embedding = np.array(query_embedding).astype(np.float32).reshape(1, -1)
        assert query_embedding.shape[1] == self.embedding_dim, "Invalid embedding dimension"

        results = []

        if node_type is not None:
            if node_type not in self.indices:
                return []
            index = self.indices[node_type]
            distances, ids = index.search(query_embedding, topk)  # ids are actual node_ids!

            for dist, id in zip(distances[0], ids[0]):
                if id == -1:  # FAISS uses -1 for missing
                    continue
                node_id = self.id_mapper.get_str(id)
                if not self.has_node(node_id):
                    continue
                node_data = self.get_node_data(node_id)
                results.append(node_data)
        else:
            raise NotImplementedError("Multiple indices not implemented yet")

        return results

    def get_nodes(self, **filters):
        return [
            (n, d['data']) for n, d in self.G.nodes(data=True)
            if all(getattr(d['data'], k, None) == v for k, v in filters.items())
        ]

    def get_edges(self, **filters):
        return [
            (u, v, d['data']) for u, v, d in self.G.edges(data=True)
            if all(getattr(d['data'], k, None) == v for k, v in filters.items())
        ]
    
    def get_node_id(
        self,
        node_type = None, 
        node_name = None,
    ):
        return self._idx.get((node_type, node_name), None)
    
    def get_node_data(self, node_id):
        if node_id in self.G.nodes and 'data' in self.G.nodes[node_id]:
            return self.G.nodes[node_id]['data']
        return None
    
    def get_edge_data(self, source_node_id, target_node_id):
        return self.G.get_edge_data(source_node_id, target_node_id)['data']

    def has_node(
        self,
        node_id = None, 
        node_type = None, 
        node_name = None,
    ):
        if self.G.has_node(node_id):
            return True
        
        if self.get_node_id(node_type, node_name):
            return True
                
        return False
    
    def has_edge(
        self,
        source_node_id = None, 
        target_node_id = None, 
        edge_type = None
    ):
        if not self.G.has_edge(source_node_id, target_node_id):
            return False
        
        if edge_type is not None:
            edge_data = self.get_edge_data(source_node_id, target_node_id)
            if edge_data.edge_type != edge_type:
                return False
        
        return True
    