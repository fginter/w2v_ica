import lwvlib
import argparse
import sklearn.decomposition
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Turn w2v to ICA')
    parser.add_argument('--w2v-model', required=True, action="store", help='W2V model (read)')
    parser.add_argument('--max-rank', type=int, default=200000, help='Max rank default %(default)d')
    parser.add_argument('--ica-model', required=True, action="store", help='ICA model (write)')
    parser.add_argument('--ica-iters', type=int, default=200, help='Number of ICA iterations')

    args = parser.parse_args()
    model=lwvlib.load(args.w2v_model,args.max_rank,args.max_rank)
    w2v_normalized=sklearn.preprocessing.normalize(model.vectors)
    f_ica=sklearn.decomposition.FastICA(max_iter=args.ica_iters)
    transformed=f_ica.fit_transform(w2v_normalized)
    np.save(args.ica_model,transformed)
    
    
