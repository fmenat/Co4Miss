# Multi-modal Co-Learning for Handling Missing Modalities


# Training

* For running Full-Co
`python
python train.py -s config/ours/fullco_crop.yaml
`


* For running Co-Miss with random version (ModDrop)
`python
python train.py -s config/ours/comissv2_crop.yaml
`


* For running Co-Miss default (all missing combinations)
`python
python train.py -s config/ours/comiss_crop.yaml
`

---

## Competitors

### Single view (for dedicated extreme missing training)

* For running Single-modality baselines
`python
python train_pool.py -s config/competitors/base_crop.yaml
`

* For running DML method
`python
python train_pool.py -s config/competitors/dml_crop.yaml
`

* For running MLA method
`python
python train_pool.py -s config/competitors/mla_crop.yaml
`

### For any type of missing view

* For running FCoM-* methods
`python
python train_multi.py -s config/competitors/fcom_crop.yaml
`

* For running EmbraceNet method
`python
python train_multi.py -s config/competitors/embr_crop.yaml
`

* For running ShaSpec method
`python
python train_multi.py -s config/competitors/shaspec_crop.yaml
`

* For running DModDrop-CML method
`python
python train_multi.py -s config/competitors/cml_crop.yaml
`

* For running MissModal method
`python
python train_multi.py -s config/competitors/missmodal_crop.yaml
`

* For running DSensD+ EO-specific method
`python
python train_multi.py -s config/competitors/dsensd_crop.yaml
`

