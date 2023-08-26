import pickle
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

data = None
search_data = None
model = None


def get_data():
    global data
    if not data:
        with open("data_img.pic", "rb") as file:
            data = pickle.load(file)
    return data


def get_search():
    global search_data
    if not search_data:
        search_data = AnnoyIndex(768, "angular")
        search_data.load("ann.ann")
    return search_data


def get_model():
    global model
    if not model:
        model = SentenceTransformer("sentence-transformers/LaBSE")
    return model


def search(search_str):
    vectors = get_model().encode([search_str])[0]
    inds = get_search().get_nns_by_vector(vectors, 5)
    res = []
    for ind in inds:
        try:
            res.append(get_data()[ind])
        except:
            pass
    return list(map(lambda x: x["link"], res))
