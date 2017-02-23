import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os
from shutil import copyfile

if __name__ == '__main__':
    model_names = ['trendfiltering', 'trendfiltering-multiscale']
    dist_types = ['gmm', 'edge_biased']
    sample_sizes = [500,1000,3000,5000,10000,15000,30000,60000]
    ntrials = 10

    K = [1,2]
    lams = [0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1.0]

    dargs = {}
    for model in model_names:
        dargs['model'] = model
        best = np.zeros((ntrials, len(sample_sizes), 5))
        best_fit = None
        for dist in dist_types:
            dargs['dist_type'] = dist
            for i in xrange(ntrials):
                dargs['train_id'] = i
                print '{model} ({dist_type}, trial {train_id})'.format(**dargs)
                for j, n in enumerate(sample_sizes):
                    print '\t\t', n
                    dargs['train_samples'] = n
                    take_any = True
                    for lam in lams:
                        dargs['lam'] = lam
                        for k in K:
                            dargs['k'] = k
                            filename = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{k}_{lam}_{train_id}'.format(**dargs)
                            if not os.path.exists(filename + '_score.csv'):
                                print 'missing {0}'.format(filename)
                                continue
                            scores = np.loadtxt(filename + '_score.csv', delimiter=',')
                            if np.isnan(scores[0]) or np.isnan(scores[1]) or np.isnan(scores[2]) or scores[0] == 0.0:
                                continue
                            if take_any or scores[0] < best[i,j,0]:
                                best[i,j,:3] = scores
                                best[i,j,3] = k
                                best[i,j,4] = lam
                                dargs['scores'] = scores
                                print 'New best: k={k} lam={lam} scores={scores}'.format(**dargs)
                                # best_fit = np.loadtxt(filename + '_fits.csv', delimiter=',')
                                best_score = scores
                                best_filename = filename
                                take_any = False
                    if not take_any:
                        outfile = 'experiments/mnist/results/{model}_{dist_type}_{train_samples}_{train_id}'.format(**dargs)
                        np.savetxt(outfile + '_score.csv', best[i,j])
                        copyfile(best_filename + '_fits.csv', outfile + '_fits.csv')




