# Deep Embedded Clustering (DEC)
This is simplified pytorch-lightning implementation of 'Unsupervised Deep Embedding for Clustering Analysis' (ICML 2016).


## Paper Review (Korean)
- [[Post] Unsupervised Deep Embedding for Clustering Analysis](https://youngerous.github.io/paper/2020/10/09/dec/)

## Environments

```markdown
- torch == 1.5.0
- torchvision == 0.6.0
- pytorch-lightning == 0.8.5
- scipy == 1.5.2
- sklearn == 0.23.2
```

## Result
- Standard K-Means Unsupervised Accuracy: 46%
- Initial Unsupervised Accuracy: 57.2%
- Trained Unsupervised Accuracy: 73.0%

p.s) Not follow paper settings strictly.

## Dataset
- MNIST in torchvision

## References
- [[Paper] Unsupervised Deep Embedding for Clustering Analysis](https://arxiv.org/abs/1511.06335)
- [[Code] dec](https://github.com/piiswrong/dec)
- [[Code] pt-dec](https://github.com/vlukiyanov/pt-dec)
