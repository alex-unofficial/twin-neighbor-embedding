from loguru import logger

import networkx as nx
import twin.dataset as data
import twin.graphmatrix as gm
import twin.embedding as emb

from twin.pipelines import batch_graph_embedding

batch_input = [{
    # small-medium size graphs (run everything)
    "common": {
        "graph": [
            ("ring_of_cliques", {"n": 10, "k": 4}),
            ("tree_of_cliques", {"h": 3, "n": 4, "k": 3}),
            "karate_club",
            "football_2023",
            ("small_world", {"n": 150, "k": 12, "p": 0.01}),
        ]
    },

    "groups": [{
        "representation": [
            "vertex",
        ],

        "embedding": [
            "spring",
            "yifanhu",
            ("forceatlas2", {
                "linlog": [False, True],
            }),
            ("kamada-kawai", {
                "mode": ['distance', 'similarity'],
            }),
            ("neato", {
                "mode": ['distance', 'similarity'],
            }),
            "spectral",
        ]
    }, {
        "representation": [
            "vertex",
            "edge",
            "twin",
        ],

        "embedding": "sgtsne",
    }]
}, {
    #large size graphs (run only faster methods)
    "common": {
        "graph": [
            ("barabasi_albert", {"n": 1000, "k": 7}),
            ("mnist", {"n_neighbors": 12}),
            ("tabula", {
                "organ": 'liver'
            }),
        ]
    },

    "groups": [{
        "representation": [
            "vertex",
        ],

        "embedding": [
            "spring",
            ("forceatlas2", {
                "linlog": [False, True],
            }),
            "spectral",
        ]
    },{
        "representation": [
            "vertex",
            "edge",
            "twin",
        ],

        "embedding": "sgtsne",
    }]
}]

batch_result = batch_graph_embedding(
    batch_input,
    progress_verbose=True,
    progress_fn=logger.info,
)
