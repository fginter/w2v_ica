import lwvlib
import argparse
import sklearn.decomposition
import numpy as np

def dimension_extremes(vectors,max_count,model):
    """Matrix words x dimensions, returns"""
    #f_ica=sklearn.decomposition.FastICA()
    transformed=np.load("ica.npy")#f_ica.fit_transform(vectors)
    transformed=sklearn.preprocessing.normalize(transformed)
    print("tr",transformed)
    print("tr",transformed.shape)
    #np.save("ica.npy",transformed)
    vectors=sklearn.preprocessing.normalize(vectors)
    top_n=np.argpartition(-np.abs(transformed),max_count,0)
    top_per_dim=(top_n.T)[:,:max_count]  #dims x max_count
    for dim in range(top_per_dim.shape[0]):
        print(dim,"   ",end=' ')
        for w in top_per_dim[dim]:
            print(model.words[w],end=' ')
        print()
            



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Turn w2v to ICA')
    parser.add_argument('--w2v-model', required=True, action="store", help='W2V model (read)')
    parser.add_argument('--max-rank', type=int, default=200000, help='Max rank default %(default)d')
    parser.add_argument('--ica-model', required=True, action="store", help='ICA model (write)')

    args = parser.parse_args()
    model=lwvlib.load(args.w2v_model,args.max_rank,args.max_rank)
    w2v_normalized=sklearn.preprocessing.normalize(model.vectors)
    f_ica=sklearn.decomposition.FastICA()
    transformed=f_ica.fit_transform(w2v_normalized)
    np.save(args.ica_model,transformed)
    
    
