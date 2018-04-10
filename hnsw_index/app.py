from flask import Flask, request, jsonify
import logging
import pyhnsw
from flasgger import Swagger


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
Swagger(app)
app.hnsw = pyhnsw.PyHNSW(b'index_data/storage.dump', b'index_data/params.dump')


@app.route('/knn', methods=['GET'])
def index():
    """
    HNSW indexer API
    ---
    tags:
      - Find k approximate neighbors

    description: Method for fast approximate neighbors search.

    parameters:
      - name: query
        in: body
        schema:
          $ref: '#/definitions/kNNRequest'

    consumes:
      - application/json

    produces:
      - application/json

    responses:
      200:
        description: Neighbors for each embedding.
        schema:
          $ref: '#/definitions/kNNResponse'

      default:
        description: Unexpected error.

    definitions:
      Embedding:
        type: array
        items:
          type: number
          format: double

      Neighbors:
        type: array
        items:
          type: integer

      kNNResponse:
        type: array
        items:
          $ref: '#/definitions/Neighbors'
        description: neighbors

      kNNRequest:
        type: object
        properties:
          query:
            type: array
            items:
              $ref: '#/definitions/Embedding'
            description: embeddings
          K:
            type: integer
            format: int32
            description: Neighbors count for each embedding.

          ef:
            type: integer
            format: int32
            description: Number of neighbors during search.
    """
    data = request.json
    q = data['query']
    K = data['K']
    ef = data['ef']

    log.info('Args: q={}, K={}, ef={}'.format(q, K, ef))
    neighbors = []
    for emb in q:
        log.info('Processing embedding: {}'.format(emb))
        emb_neighbors = app.hnsw.knn_search(emb, K, ef)
        log.info('Embedding neighbors: {}'.format(emb_neighbors))
        neighbors.append(emb_neighbors)

    return jsonify(neighbors)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
