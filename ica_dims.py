import lwvlib
import argparse
import sklearn.decomposition
import numpy as np

def dimension_extremes(vectors,max_count,model):
    """Matrix words x dimensions, returns"""
    vectors=sklearn.preprocessing.normalize(vectors)
    top_n=np.argpartition(-np.abs(vectors),max_count,0)
    top_per_dim=(top_n.T)[:,:max_count]  #dims x max_count
    for dim in range(top_per_dim.shape[0]):
        print(dim,"   ",end=' ')
        for w in top_per_dim[dim]:
            print(model.words[w],end=' ')
        print()
            



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='ICA postprocessing for w2v')
    parser.add_argument('--w2v-model', required=True, action="store", help='W2V model')
    parser.add_argument('--ica-model', required=True, action="store", help='ICA model')
    parser.add_argument('--max-rank', type=int, default=200000, help='Max rank default %(default)d')
    
    args = parser.parse_args()
    model=lwvlib.load(args.w2v_model,args.max_rank,args.max_rank)
    ica_matrix=np.load(args.ica_model)
    dimension_extremes(ica_matrix,15,model)

    
