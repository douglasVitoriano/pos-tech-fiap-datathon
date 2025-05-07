class MergeFrames:
    def __init__(self, df1, df2):
        self.df1 = df1
        self.df2 = df2

    def merge(self, left_on, right_on, how='inner'):
        return self.df1.merge(self.df2, left_on=left_on, right_on=right_on, how=how)