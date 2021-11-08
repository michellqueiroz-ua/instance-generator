from input_json import input_json
from streamlit import caching

if __name__ == '__main__':

    caching.clear_cache()

    input_json('../examples/03_example.json')