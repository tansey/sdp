import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import os
from shutil import copyfile

if __name__ == '__main__':
    dargs = {}
    dargs['model'] = 'skd'
    dargs['dist_type'] = 'edge_biased'
    dargs['nbins'] = 2500
    dargs['train_samples'] = 2500
    dargs['k'] = 1
    
    neighbor_radius = [1,3,5,10,25,50]
    lams = [0.0001,0.001,0.01,0.1,1.0,10.0]
    
    for train_id in xrange(10):
        dargs['train_id'] = train_id
        for radius in neighbor_radius:
            dargs['neighbor_radius'] = radius
            print 'Neighborhood radius: {neighbor_radius}'.format(**dargs)
            best_score = None
            for lam in lams:
                dargs['lam'] = lam
                filename = 'experiments/mnist/results/{model}_{dist_type}_{nbins}_{train_samples}_{k}_{lam}_{neighbor_radius}_{train_id}'.format(**dargs)
                if not os.path.exists(filename + '_score.csv'):
                    print 'missing {0}'.format(filename)
                    continue
                scores = np.loadtxt(filename + '_score.csv', delimiter=',')
                if np.isnan(scores[0]) or np.isnan(scores[1]) or np.isnan(scores[2]) or scores[0] == 0.0:
                    continue
                if best_score is None or scores[0] < best_score[0]:
                    best_score = scores
            if best_score is not None:
                outfile = 'experiments/mnist/results/{model}_{dist_type}_{nbins}_{train_samples}_{neighbor_radius}_{train_id}'.format(**dargs)
                np.savetxt(outfile + '_score.csv', best_score)




