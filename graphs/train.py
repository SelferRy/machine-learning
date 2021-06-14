from make_dataset import train, val, test

X, Y = {}, {}
X["train"], Y["train"] = train
X["val"], Y["val"] = val
X["test"], Y["test"] = test
