class MLNode:
    def __init__(self, param, paramCol, group):
        self.param = param
        self.paramCol = paramCol
        self.group = group
    
    def __repr__(self):
        if self.param != "PURE" or self.paramCol != "PURE":
            return f"Parameter: {self.param} Col: {self.paramCol}\n       Group: {self.group}"
        else:
            return  f"NODE IS PURE! {self.group}"
    
    