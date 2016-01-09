# -*- coding: utf-8 -*-
"""
 Simple function for processing results
"""

import matplotlib as plt

def simple_plot_wac_curves(monitor_outputs, batch_size):
    """ Plot WAC score monitor outputs """
    plt.clf()
    unc_wac = monitor_outputs['wac_score_labeled']
    unc_wac_unlabeled = monitor_outputs['wac_score_unlabeled']
    unc_wac_valid = monitor_outputs['wac_score_valid']

    t = np.arange(len(unc_wac)) * batch_size
    p1 = plt.plot(t, unc_wac, 'r')
    p2 = plt.plot(t, unc_wac_valid, 'g')
    p3 = plt.plot(t, unc_wac_unlabeled, 'b')
    plt.legend((p1[0], p2[0], p3[0]), ('WAC_labeled', 'WAC valid', 'WAC not seen'), loc='best')
    plt.title('Models Accuracy')
    plt.show()
