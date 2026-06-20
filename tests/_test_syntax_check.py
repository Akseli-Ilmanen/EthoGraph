import ast
for p in ["ethograph/gui/widgets_data.py", "ethograph/gui/widgets_changepoints.py"]:
    ast.parse(open(p, encoding="utf-8").read())
print("syntax OK")
