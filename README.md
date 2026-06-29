# wyoloservice2_mcp

Servidor de Model Context Protocol (MCP) para el clúster Train Service 2 / NeuralForgeAI. Permite a los agentes de Inteligencia Artificial conectarse nativamente para inspeccionar y controlar los entrenamientos de YOLO.

## Instalación

Instala el paquete y sus dependencias (FastMCP) usando pip:

```bash
pip install -e .
```

## Uso con un Cliente MCP (ej. Claude Desktop o Antigravity)

Añade este servidor a tu archivo de configuración MCP (por ejemplo, en `mcp_servers.json`):

```json
{
  "mcpServers": {
    "neuralforge-mcp": {
      "command": "wyolo-mcp",
      "env": {
        "NEURALFORGE_API_URL": "http://192.168.1.68:8000"
      }
    }
  }
}
```

## Herramientas Expuestas
1. **`get_cluster_status`**: Comprueba si la API y Celery están online.
2. **`get_study_details(study_id)`**: Recupera métricas, progresos y el worker activo de un estudio.
3. **`cancel_study(study_id)`**: Fuerza la cancelación de un estudio y mata los contenedores efímeros.
4. **`launch_training(config)`**: Envía un nuevo estudio YOLO a la cola.
5. **`check_dataset_path(path)`**: Verifica localmente si un volumen o ruta de dataset existe antes de lanzar entrenamientos para evitar fallos.