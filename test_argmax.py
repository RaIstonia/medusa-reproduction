import jittor as jt
a = jt.array([[1, 2, 3], [4, 5, 6]])
res = jt.argmax(a, dim=-1)
print(f"Type: {type(res)}")
print(f"Value: {res}")

if isinstance(res, tuple):
    print(f"Tuple len: {len(res)}")
    print(f"Item 0 type: {type(res[0])}")
    if len(res) > 1:
        print(f"Item 1 type: {type(res[1])}")

