import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from shutil import copyfile
import os

if __name__ == '__main__':
    model_names = ['lmm', 'gmm']
    #dist_types = ['gmm', 'discontinuous_gmm', 'edge_biased']
    dist_types = ['gmm', 'edge_biased']
    sample_sizes = [500,1000,3000,5000,10000,15000,30000,60000]
    ntrials = 10

    num_components = [1, 3, 5, 10, 20]
    dargs = {}
    for model in model_names:
        dargs['model'] = model
        best = np.zeros((ntrials, len(sample_sizes), 4))
        best_fit = None
        for dist in dist_types:
            dargs['dist_type'] = dist
            for i in xrange(ntrials):
                print i
                dargs['train_id'] = i
                for j, n in enumerate(sample_sizes):
                    print '\t', n
                    dargs['train_samples'] = n
                    take_any = True
                    for k in num_components:
                        dargs['num_components'] = k
                        filename = 'experiments/mnist/results/{model}_{num_components}_{dist_type}_{train_samples}_{train_id}'.format(**dargs)
                        if not os.path.exists(filename + '_score.csv'):
                            print 'missing {0}'.format(filename)
                            continue
                        scores = np.loadtxt(filename + '_score.csv', delimiter=',')
                        if np.isnan(scores[0]) or np.isnan(scores[1]) or np.isnan(scores[2]):
                            continue
                        if take_any or scores[0] < best[i,j,0]:
                            best[i,j,:3] = scores
                            best[i,j,3] = k
                            # best_fit = np.loadtxt(filename + '_fits.csv', delimiter=',')
                            best_score = scores
                            best_filename = filename
                            take_any = False
                    outfile = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{train_id}'.format(**dargs)
                    np.savetxt(outfile + '_score.csv', best[i,j])
                    copyfile(best_filename + '_fits.csv', outfile + '_fits.csv')




