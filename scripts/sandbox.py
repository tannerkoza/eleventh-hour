import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import defaultdict

N = 4


@dataclass
class Test:
    a: float = 1.0
    b: float = 2.0


my_dict = defaultdict(lambda: [])

for _ in range(100):
    random_key1 = str(np.random.randint(0, 10))
    random_key2 = str(np.random.randint(10, 20))

    my_dict[(random_key1, random_key2)].append(Test())

df = pd.DataFrame.from_dict(
    dict([(key, pd.Series(value)) for key, value in my_dict.items()])
)

print("")
