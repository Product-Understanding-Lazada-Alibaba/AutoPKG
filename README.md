# APKG code and data repo

### This is the minimum reproducible version extracted from internal experimental code.

## DATA:

This repository includes four data files located in the `data/` directory, derived from the **Lazada** e-commerce platform (Philippines region).

---

### 1. `lazada_autopkg_kg_nodes.csv` — Knowledge Graph Nodes

Contains all nodes (entities) in the product knowledge graph. There are three types of nodes: **Product**, **Type** (product category), and **Key** (attribute key).

| Column | Description |
| :--- | :--- |
| `node_id` | Unique identifier of the node (e.g., `T00001` for Type, `K00023` for Key, numeric ID for Product). |
| `node_type` | Type of the node: `Product`, `Type`, or `Key`. |
| `node_name` | Canonical name of the node. For Product nodes, this is the `product_id` (e.g., `8576940.PH`). |
| `description` | A natural-language description of the node, generated or extracted by LLM. |
| `examples` | Representative example values associated with the node. |
| `synonyms` | A JSON list of alternative names or aliases for the node. |
| `created_at` | Timestamp when the node was first created. |
| `updated_at` | Timestamp of the last update to the node. |
| `ds` | Data partition date string (e.g., `20251230`). |
| `model_name` | The model or pipeline version used to generate this node. |

---

### 2. `lazada_autopkg_kg_edges.csv` — Knowledge Graph Edges

Contains all directed edges (relationships) between nodes in the knowledge graph. The primary relationship is `has_key`, connecting a **Type** node to a **Key** node, indicating that a product category possesses a certain attribute.

| Column | Description |
| :--- | :--- |
| `source_node_id` | The `node_id` of the source node (typically a Type node). |
| `target_node_id` | The `node_id` of the target node (typically a Key node). |
| `edge_type` | The type of relationship (e.g., `has_key`). |
| `description` | A natural-language description of what this attribute means in context. |
| `examples` | Representative example values for this attribute under the given product type. |
| `ranking` | An importance ranking score indicating how significant this attribute is for the product type. |
| `created_at` | Timestamp when the edge was first created. |
| `updated_at` | Timestamp of the last update to the edge. |
| `ds` | Data partition date string (e.g., `20251230`). |
| `model_name` | The model or pipeline version used to generate this edge. |

---

### 3. `lazada_autopkg_product_data_file_path.csv` — Product Data (Image File Paths)

Contains structured product information for ~36,000 Lazada products. In this version, **product images are stored on Google Drive** and referenced by their **relative file paths** within the drive folder. 

Link: https://drive.google.com/file/d/16kIB89v7DwxBESSIZS5WeV28AVjinRTi/view?usp=sharing

The folder name for each product corresponds to its `product_id`, and images inside the folder are named with numbers (e.g., `1.jpg`, `2.jpg`). To keep the total dataset size manageable, **each product is limited to at most 3 images**.

> 📦 The image files can be accessed via the Google Drive link provided separately.

| Column | Description |
| :--- | :--- |
| `product_id` | Unique product identifier (e.g., `53156.PH`). Also serves as the folder name in Google Drive. |
| `product_name` | The display name / title of the product. |
| `highlight` | Short highlight bullet points of the product features (may be `null`). |
| `description` | Full text description of the product. |
| `specifications` | Structured specification text of the product (may be `null`). |
| `images` | A JSON list of image file names (e.g., `["1.jpg", "2.jpg"]`) stored under the `product_id` folder in Google Drive. At most 3 images per product. |
| `description_images` | A JSON list of image file names used within the product description section. |

---

### 4. `lazada_autopkg_product_data_url.csv` — Product Data (Image URLs)

Contains the same structured product information as above, but in this version **product images are stored as public URLs** pointing directly to Lazada's CDN. Users can access the images directly via these links without downloading any files.

| Column | Description |
| :--- | :--- |
| `product_id` | Unique product identifier (e.g., `53156.PH`). |
| `product_name` | The display name / title of the product. |
| `highlight` | Short highlight bullet points of the product features (may be `null`). |
| `description` | Full text description of the product. |
| `specifications` | Structured specification text of the product (may be `null`). |
| `images` | A JSON list of public image URLs for the main product images. |
| `description_images` | A JSON list of public image URLs used within the product description section. |

---

## CODE:
**KGD** is an asynchronous, distributed pipeline for building and maintaining product knowledge graphs. It leverages Large Language Models (LLMs) for entity resolution, property enrichment, and relationship extraction, utilizing FAISS for efficient semantic similarity search and NetworkX for graph structure management.

The system is designed to run on large-scale datasets stored in **ODPS (MaxCompute)**, supporting multi-GPU parallel processing with fine-grained locking mechanisms to ensure data consistency during concurrent updates.

## 🌟 Key Features

*   **Async Concurrency**: Built on `asyncio` with `vllm` for high-throughput non-blocking inference.
*   **Distributed Processing**: Supports multi-node/multi-GPU execution via environment variables (`RANK`, `WORLD_SIZE`).
*   **Entity Resolution**: Intelligent agent decides whether to **ADD**, **MERGE**, **REPLACE**, or **DISCARD** nodes based on strict synonymy policies.
*   **Semantic Search**: Uses FAISS (`IndexIDMap`) for fast nearest-neighbor lookups to prevent duplicate entities.
*   **Dynamic Enrichment**: Automatically generates descriptions and examples for nodes lacking properties using LLMs.
*   **ODPS Integration**: Native support for reading from and writing to Alibaba Cloud ODPS tables.
*   **Fine-Grained Locking**: Prevents race conditions when multiple workers process the same entity type or name simultaneously.

## 🏗 Architecture

The system consists of several core components:

1.  **Orchestrator (`main.py`)**: Handles argument parsing, dataset streaming, and initializes the `KGD` agent.
2.  **KGD Agent (`kgd.py`)**: The brain of the operation.
    *   **Worker Loop**: Pulls items from a queue, acquires locks, runs inference, and queues update requests.
    *   **Updater Loop**: Sequentially applies changes to the in-memory graph (`SimpleKG`) to ensure consistency.
    *   **Distributed Coordination**: Master node exposes an HTTP API for worker nodes to request new Node IDs safely.
3.  **Engines (`engines.py`)**: Wrappers around `vllm` for asynchronous text generation (Agent) and embedding creation.
4.  **Graph Store (`simple_kg.py`)**: In-memory graph using `networkx` for topology and `faiss` for vector indexing.
5.  **Data Models (`data.py`)**: Pydantic models defining Nodes, Edges, and Logs with automatic datetime formatting.

## 📋 Prerequisites

*   Python 3.9+
*   GPU(s) with CUDA support
*   Access to Alibaba Cloud ODPS (Access ID, Key, Endpoint)
*   Required Python packages (see `requirements.txt` below)

### Dependencies

You will need to install the following libraries:

```bash
pip install asyncio tqdm transformers datasets pydantic vllm faiss-cpu networkx pandas odps-python-client openlm_hub fastapi uvicorn python-dotenv torch numpy
```

*(Note: `faiss-gpu` is recommended if running on GPU for index operations, though `faiss-cpu` works for smaller graphs).*

## ⚙️ Configuration

Set up your environment variables for ODPS access. You can use a `.env` file:

```bash
ACCESS_ID=your_access_id
ACCESS_KEY=your_access_key
PROJECT=your_project_name
ODPS_ENDPOINT=your_odps_endpoint
MASTER_ADDR=127.0.0.1
MASTER_PORT=8000
```

## 🚀 Usage

### Basic Single-GPU Run

Run the pipeline for a specific task (e.g., `type`, `key`, or `value`):

```bash
python main.py \
    --input "odps://project/tables/source_data" \
    --node_output "odps://project/tables/nodes_out/partition=date=20231027" \
    --edge_output "odps://project/tables/edges_out/partition=date=20231027" \
    --log_table "odps://project/tables/logs_out/partition=date=20231027" \
    --emb_model_name "path/to/embedding/model" \
    --agent_model_name "path/to/llm/model" \
    --task "type" \
    --save_steps 5000
```

### Distributed Multi-GPU Run

To run across multiple GPUs on a single machine or across a cluster, set the `RANK` and `WORLD_SIZE` environment variables.

**Example: 4 GPUs on one machine**

```bash
# Terminal 1 (Rank 0 - Master)
RANK=0 WORLD_SIZE=4 python main.py ... [args]

# Terminal 2 (Rank 1)
RANK=1 WORLD_SIZE=4 python main.py ... [args]

# Terminal 3 (Rank 2)
RANK=2 WORLD_SIZE=4 python main.py ... [args]

# Terminal 4 (Rank 3)
RANK=3 WORLD_SIZE=4 python main.py ... [args]
```

*Note: Rank 0 acts as the master coordinator and hosts the HTTP server for ID generation if `--task value` is used.*

## 📂 Arguments

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--input` | ODPS path to the input dataset containing raw entities. | `None` |
| `--node_input` | (Optional) ODPS path to existing nodes to warm-start the graph. | `None` |
| `--node_output` | ODPS path to save the final nodes. | `None` |
| `--edge_input` | (Optional) ODPS path to existing edges. | `None` |
| `--edge_output` | ODPS path to save the generated edges. | `None` |
| `--log_table` | ODPS path to save inference logs (prompts, actions, thoughts). | `None` |
| `--emb_model_name` | Path or HF repo ID for the embedding model. | `None` |
| `--agent_model_name` | Path or HF repo ID for the LLM agent model. | `None` |
| `--task` | Task type: `type`, `key`, or `value`. Determines ID prefix and logic. | `None` |
| `--save_steps` | Frequency (in rows) to push intermediate results to ODPS. | `100000` |

## 🧠 Logic Flow

1.  **Ingestion**: Streams rows from ODPS. Each row contains a `node_name`, `node_type`, `properties`, and potential `edges`.
2.  **Queueing**: Items are placed into an async queue.
3.  **Processing (Worker)**:
    *   Checks if the current rank owns this `node_type` (hash-based sharding).
    *   Acquires a lock for `(node_type, node_name)`.
    *   **Exact Match**: Checks local index. If found, flags for property update.
    *   **Enrichment**: If description missing, calls LLM to generate it.
    *   **Embedding**: Generates vector representation.
    *   **Candidate Check**: If no exact match, searches FAISS for similar nodes. Sends context to LLM Agent to decide action (**ADD**, **MERGE**, **REPLACE**).
4.  **Updating (Updater)**:
    *   Receives the decision sequentially.
    *   Updates the `networkx` graph and `faiss` index.
    *   Handles edge creation (resolving "this" placeholders to actual Node IDs).
5.  **Persistence**: Periodically saves the state back to ODPS.

## 📝 Data Schema

### Input Data
Expected columns in the input ODPS table:
*   `node_name`: String
*   `node_type`: String
*   `description`: String (Optional)
*   `examples`: String (Optional)
*   `edges`: JSON String (List of edge objects)

### Output Nodes
*   `node_id`: Unique identifier (e.g., `T00001`, `K00023`).
*   `node_name`: Canonical name.
*   `synonyms`: List of alternative names.
*   `properties`: Dict containing `description`, `examples`, etc.
*   `create_time`, `update_time`: Timestamps.

### Output Logs
Contains the full trace of the LLM decision process:
*   `prompt`: The formatted prompt sent to the agent.
*   `thought`: The reasoning extracted from the LLM output.
*   `action`: The final decision (e.g., `MERGE T00001`).

## 🔒 Concurrency & Safety

*   **Type Locks**: `asyncio.Lock` per `node_type` ensures that candidate checking and updates for a specific category happen in a consistent order.
*   **Name Locks**: Fine-grained locks per `(type, name)` prevent duplicate processing of the same entity by different workers.
*   **Master-Worker Sync**: For the `value` task, workers communicate with Rank 0 via HTTP to guarantee unique Node ID generation across the cluster.

## 🛠 Debugging

Set `--debug_rows N` to print the first `N` prompts and actions to stdout in a formatted box. This helps verify the LLM's reasoning and adherence to the "Strict Synonym Policy".
