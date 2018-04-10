from flask import Flask, request, jsonify
import logging
import pyhnsw


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = Flask(__name__)
app.hnsw = pyhnsw.PyHNSW(b'index_data/storage.dump', b'index_data/params.dump')


@app.route('/knn', methods=['GET'])
def index():
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
