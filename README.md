# Frenet
This repository provides a suite to transform coordinates from the **Cartesian** to the **Frenet** frame. Besides the basic transformation, we also provide complete extensions for the Argoverse dataset.

## Dependency
For basic usage, you only need `numpy` in your Python environment, and `matplotlib` if you require visualization functions.

For those working with the Argoverse dataset, please first follow the instructions provided by the [Argoverse API](https://github.com/argoverse/argoverse-api) to install the `argoverse` package. Then, proceed to install the following requirements:
```
pip install -r requirements.txt
```

## Quick Start
```python
from frenet import cartesian_to_frenet

s, d, direction = cartesian_to_frenet(x, y, ref_path)
```
where `x, y` are numpy arrays with the shape of `(n,)`, indicating the trajectory coordinates in the Cartesian frame, and `ref_path` is a numpy array with the shape of `(m, 2)`, indicating the coordinates of the reference path in the Cartesian frame.

The output `s, d` are numpy arrays with the shape of `(n,)`, indicating the trajectory coordinates in the Frenet frame, and `direction` is a boolean array with the shape of `(n,)`. `True` indicates that the point is on the left of the reference path, while `False` indicates that the point is on the right.

Please refer to the `\demo` folder, for more demonstrations, such as *Work on Argoverse*.

## TODO
- [ ] Include more text descriptions/comments in the demo.
- [ ] Complete the docstrings of the functions.
- [ ] Provide more demonstrations about the functions.
- [ ] Rebuild the function to transform coordinates from the Frenet frame to the Cartesian frame.

## Paper and Citation
If you think this repository is helpful, please cite our paper [Improving the Generalizability of Trajectory Prediction Models with Frenet-Based Domain Normalization](https://arxiv.org/abs/2305.17965).
```
@misc{ye2023improving,
      title={Improving the Generalizability of Trajectory Prediction Models with Frenet-Based Domain Normalization}, 
      author={Luyao Ye and Zikang Zhou and Jianping Wang},
      year={2023},
      eprint={2305.17965},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```