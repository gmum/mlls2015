from experiments.analyze import *

if __name__ == "__main__":

    # for compound in ["5-HT7", "5-HT2c", "5-HT2a", "5-HT6", "5-HT1a", "d2",
    #                  "5-HT7_DUDs", "5-HT2c_DUDs", "5-HT2a_DUDs", "5-HT6_DUDs", "5-HT1a_DUDs", "d2_DUDs"]:
    for compound in ['5-HT2a', '5-HT2c', '5-HT7']:
        for fingerprint in ['Pubchem', 'Ext', 'Klek']:
            print "Starting %s %s" % (compound, fingerprint)
            results_dir = os.path.join(RESULTS_DIR, 'QGB_SUB', "SVM", compound, fingerprint)

            process_results(results_dir, 'wac_score_valid_score_auc', essential_only=True)
            process_results(results_dir, 'wac_score_valid_aleph_score_auc', essential_only=True)

            if "DUDs" in compound:
                process_results(results_dir, 'wac_score_valid_score-duds_auc', essential_only=True)
                process_results(results_dir, 'wac_score_valid_aleph_score-duds_auc', essential_only=True)



