# Missingness-to-detection findings

Paired masks were evaluated on five held-out turbines after fitting the normal-behaviour model on clean turbines 0-2. The clean warning-vs-healthy ROC-AUC was 0.980, with an 11.0% healthy-day false-alarm rate.

Among corrupted settings, **linear** was strongest: random missingness at 10% retained ROC-AUC 0.975. Linear interpolation also had the lowest reconstruction RMSE in every tested pattern/severity combination. Its ROC-AUC fell from 0.980 clean to 0.962 under 30% random missingness and 0.824 under 30% sensor dropout. Healthy-day false alarms rose to 63.4% and 39.7%, respectively. Event detection saturated at 100%, so ROC-AUC and false alarms are the informative robustness outcomes. This controlled synthetic result does not claim that linear interpolation is universally optimal on CARE or operational SCADA.
