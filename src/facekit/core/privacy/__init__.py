"""Privacy audit of synthetic faces against the real images that trained them.

Port of the standalone ``face_audit`` package used for the manuscript's
privacy assessment, with its Hydra configuration and GMDB-specific cohort
mapping replaced by three folder roots (train / held-out / synthetic) that
share cohort subfolder names.
"""
