## Supervised Learning

- Task:
	- mapping $f$ from inputs $x \in \mathcal{X}$ to outputs $y \in \mathcal{Y}$. 
	- given some characteristics (often numerical), predicting some value(s).
	- we call the inputs $x$ features.
		- fixed-dimensional vector of numbers (for e.g: height and weight of a person, or the pixels of an image).
	- we call the output $y$ label or target.
- Experience: Training Dataset
	- $N$ input-output pairs $D = \{x_i, y_i\}_{i = 1}^N$
	- $N$: sample size
- Performance: depends on the type of output we predict.


- Training Dataset $D = \{x_i, y_i\}_{i = 1}^N$
	- $x_i \in \mathbb{R}^d, y_i \in \mathbb{R}$ for regression
	- $x_i \in \mathbb{R}^d, y_i \in \{0, 1, \dots, k - 1\}$ for classification with $k$ classes
- Model:
	- $f(x_i)$ predicts some value for every object.
- Loss function $\mathcal{L}(D, f)$ that should be minimized.