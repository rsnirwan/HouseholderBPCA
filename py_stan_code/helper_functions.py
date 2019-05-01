import pystan
import pickle
from hashlib import md5


# from: http://pystan.readthedocs.io/en/latest/avoiding_recompilation.html - modified
def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'py_stan_code/cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'py_stan_code/cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm
