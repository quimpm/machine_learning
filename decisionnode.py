class decisionnode:
    
    def __init__(self, col=-1, value=None, results=None, tb=None, fb=None):
        self.col=col
        self.value=value
        self.results=results
        self.tb=tb
        self.fb=fb

    def toString(self):
        return '['+str(self.col)+','+str(self.value)+','+str(self.results)+','+str(self.tb)+','+str(self.fb)+']'