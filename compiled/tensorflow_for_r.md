Tensorflow for R
================

The aim is to go through some of the basics of using `tensorflow` for
auto-differentiation.

``` r
# Just some setup
library(dplyr)
```

### Installation

I have been able to get everything to work via using `conda` and the
standard install script. The official page for this is found
here\[<https://tensorflow.rstudio.com/installation/>\].

Note: When using the installation with `conda`, I had some major issues
when it was not downloaded into the standard user location, i.e.,
`/home/<user>/anaconda3`.

What I did
    was:

``` r
install.packages("tensorflow")
```

    ## Installing package into '/home/asoen/R/x86_64-pc-linux-gnu-library/3.6'
    ## (as 'lib' is unspecified)

``` r
library(tensorflow)
install_tensorflow("conda")
```

    ## 
    ## Installation complete.

``` r
# Also included and useful
library(keras)
```

A new conda environment will be made for tensorflow for R:
`r-reticulate`. You should already have this if you are using the
`reticulate` package.

### Basic Constant/Variable

There seens to be two main datatypes: - The constant `tf$constant`; -
The variable `tf$Variable`.

These are methods for defining tensors. We can define tensors simply by
passing R lists:

``` r
r_list <- list(c(1, 2, 3), c(4, 5, 6))
const_tensor <- tf$constant(r_list)
var_tensor <- tf$Variable(r_list)

print(const_tensor)
```

    ## tf.Tensor(
    ## [[1. 2. 3.]
    ##  [4. 5. 6.]], shape=(2, 3), dtype=float32)

``` r
print(var_tensor)
```

    ## <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
    ## array([[1., 2., 3.],
    ##        [4., 5., 6.]], dtype=float32)>

One of the nice inherited properties from Python is the indexing
notation.

``` r
print(const_tensor[, `::-2`])
```

    ## tf.Tensor(
    ## [[3. 1.]
    ##  [6. 4.]], shape=(2, 2), dtype=float32)

``` r
print(var_tensor[, `::-2`])
```

    ## tf.Tensor(
    ## [[3. 1.]
    ##  [6. 4.]], shape=(2, 2), dtype=float32)

For non-R-native indexing notation, the “tick”s are required. Also
notice that the resulting tensors are of the constant type. Full
information can be found
here\[<https://tensorflow.rstudio.com/guide/tensorflow/tensors/>\].

There are some important differences for the two types:

1.  I haven’t been able to find a method of doing assignment for
    constant tensors (as per its name-sake). However, we can do this for
    variables:

<!-- end list -->

``` r
print(var_tensor)
```

    ## <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
    ## array([[1., 2., 3.],
    ##        [4., 5., 6.]], dtype=float32)>

``` r
var_tensor[1, 2]$assign(-1)
```

    ## <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    ## array([[ 1., -1.,  3.],
    ##        [ 4.,  5.,  6.]], dtype=float32)>

``` r
print(var_tensor)
```

    ## <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
    ## array([[ 1., -1.,  3.],
    ##        [ 4.,  5.,  6.]], dtype=float32)>

One work around for constant vectors is the use of indicator
functions/one-hot-vectors:

``` r
indicators_1 <- tf$one_hot(as.integer(c(0, 1)), as.integer(2))
indicators_2 <- tf$one_hot(as.integer(c(0, 1, 2)), as.integer(3))
idx_1 <- tf$reshape(indicators_1[1, ], shape(1, -1))
idx_2 <- tf$reshape(indicators_2[2, ], shape(1, -1))

position_indicator <- tf$transpose(idx_1) * idx_2
print(position_indicator)
```

    ## tf.Tensor(
    ## [[0. 1. 0.]
    ##  [0. 0. 0.]], shape=(2, 3), dtype=float32)

``` r
const_tensor <- const_tensor + position_indicator * ((-1) - const_tensor[1, 2])
print(const_tensor)
```

    ## tf.Tensor(
    ## [[ 1. -1.  3.]
    ##  [ 4.  5.  6.]], shape=(2, 3), dtype=float32)

NOTE: that in `tf$one_hot` we switch from R indexing (start from 1) to
Python index. This happens for some of the functions in `tf$<function>`.
Check the **Python** tensorflow documentation when working with these:
here\[<https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/one_hot>\].

Please let me know if you find a better way\!

2.  In variable tensors, we cannot use assign when calculating
    gradients\! This one is an important one. We will go through
    automatic differentiation immediately below now.

### Automatic Differentiation

There are a few important functions which we need: -
`tf$GradientTape()`: keeps track of computation and calculate gradients;
- `t$watch(x)` where `t` is the tape: informs the tape to keep track of
computation for constant tensor `x`.

Here is a simple example:

``` r
f1 <- function(x, y) {
  acc <- 0
  for (i in 1:length(y)) {
    acc <- x[1] * y[i] + log(x[2] * y[i])
  }
  return(acc)
}

grad.f1 <- function(x, y) {
  with(tf$GradientTape() %as% t, {
    t$watch(x)
    val <-f1(x, y)
  })
  return(t$gradient(val, x))
}

input <- c(5, 2)
y_input <- c(2, 3, 4, 5)

print(f1(input, y_input))
```

    ## [1] 27.30259

``` r
print(grad.f1(tf$Variable(input), y_input))
```

    ## tf.Tensor([5.  0.5], shape=(2,), dtype=float32)

``` r
print(grad.f1(tf$constant(input), y_input))
```

    ## tf.Tensor([5.  0.5], shape=(2,), dtype=float32)

``` r
print(grad.f1(input, y_input))  # Notice this gives an error
```

    ## Error in py_call_impl(callable, dots$args, dots$keywords): ValueError: Passed in object of type <class 'float'>, not tf.Tensor
    ## 
    ## Detailed traceback: 
    ##   File "/home/asoen/anaconda3/envs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/eager/backprop.py", line 850, in watch
    ##     type(t)))

Note that the result will be a tensor of constant type, and by default
is `tf$float32`. Also note that when computing the gradient, tensorflow
becomes unhappy when our input is not in a tensorflow format. We can
make a quick wrapper around the gradient function as follows:

``` r
r_grad.f1 <- function(x, y) {
  x <- tf$convert_to_tensor(x)
  return(grad.f1(x, y))
}

r_grad_r.f1 <- function(x, y) {
  x <- tf$convert_to_tensor(x)
  return(grad.f1(x, y)$numpy())
}


print(r_grad.f1(input, y_input))  # Now it is happy :)
```

    ## tf.Tensor([5.  0.5], shape=(2,), dtype=float32)

``` r
print(r_grad_r.f1(input, y_input))  # Converts back to R for demonstration below
```

    ## [1] 5.0 0.5

Lets compare the gradients now:

``` r
library(numDeriv)
library(nloptr)

print(grad(function(x) f1(x, 2), input))
```

    ## [1] 2.0 0.5

``` r
check.derivatives(.x = input, func = function(x) f1(x, 2), func_grad = function(x) r_grad_r.f1(x, 2))
```

    ## Derivative checker results: 0 error(s) detected.

    ## 
    ##   grad_f[ 1 ] = 2e+00 ~ 2e+00   [0e+00]
    ##   grad_f[ 2 ] = 5e-01 ~ 5e-01   [0e+00]

    ## $analytic
    ## [1] 2.0 0.5
    ## 
    ## $finite_difference
    ## [1] 2.0 0.5
    ## 
    ## $relative_error
    ## [1] 0 0
    ## 
    ## $flag_derivative_warning
    ## [1] FALSE FALSE

Pre-good right? Well lets try a different function. Here are the inputs:

``` r
x_input <- c(1, 2, 3.5, 1, 2)
y_input <- c(1.5, 2.2, 3, 1, 4)
t <- 2
```

Here are the outputs:

``` r
f2 <- function(t, xs, ys) {
  acc <- tf$zeros_like(xs)
  for (i in 1:length(xs)) {
    mask <- tf$cast(ys < t, tf$float32)
    acc <- acc + (t - xs[i] * log(t)) * mask
  }
  return(acc %>% k_sum)
}

f2_t <- function(t) return(f2(t, x_input, y_input))

grad.f2_t <- function(t) {
  with(tf$GradientTape() %as% tape, {
    tape$watch(t)
    val <-f2_t(t)
  })
  return(tape$gradient(val, t))
}

r_grad_r.f2_t <- function(t) {
  t <- tf$convert_to_tensor(t)
  return(grad.f2_t(t)$numpy())
}
```

However, there is a bit of a problem when we try and use optimisers in R
with these
gradients.

``` r
print(f2_t(t))
```

    ## tf.Tensor(6.8302035, shape=(), dtype=float32)

``` r
print(r_grad_r.f2_t(t))
```

    ## [1] 0.5

``` r
print(grad(function(t) f2_t(t)$numpy(), t))
```

    ## [1] 0.5190869

``` r
check.derivatives(.x = t, func = function(t) f2_t(t)$numpy(), func_grad = r_grad_r.f2_t)
```

    ## Derivative checker results: 1 error(s) detected.

    ## 
    ## * grad_f[ 1 ] = 5e-01 ~ 0e+00   [5e-01]

    ## $analytic
    ## [1] 0.5
    ## 
    ## $finite_difference
    ## [1] 0
    ## 
    ## $relative_error
    ## [1] 0.5
    ## 
    ## $flag_derivative_warning
    ## [1] TRUE

So what happened here? Basically, there is a problem when switching
between tensorflow’s float32 type and R’s double. Lets try to make
everything `tf$float64`.

``` r
f3 <- function(t, xs, ys) {
  acc <- tf$zeros_like(xs)
  for (i in 1:length(xs)) {
    mask <- tf$cast(ys < t, tf$float64)
    acc <- acc + (t - xs[i] * log(t)) * mask
  }
  return(acc %>% k_sum)
}

f3_t <- function(t) return(f3(t, tf$constant(x_input, tf$float64), tf$constant(y_input, tf$float64)))

grad.f3_t <- function(t) {
  with(tf$GradientTape() %as% tape, {
    tape$watch(t)
    val <-f3_t(t)
  })
  return(tape$gradient(val, t))
}

r_grad_r.f3_t <- function(t) {
  t <- tf$convert_to_tensor(t)
  return(grad.f3_t(t)$numpy())
}
```

Now to try
    again:

``` r
print(f3_t(t))
```

    ## tf.Tensor(6.830203569361039, shape=(), dtype=float64)

``` r
print(r_grad_r.f3_t(t))
```

    ## Error in py_call_impl(callable, dots$args, dots$keywords): InvalidArgumentError: cannot compute Less as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:Less]
    ## 
    ## Detailed traceback: 
    ##   File "/home/asoen/anaconda3/envs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_math_ops.py", line 5376, in less
    ##     _six.raise_from(_core._status_to_exception(e.code, message), None)
    ##   File "<string>", line 3, in raise_from

``` r
print(grad(function(t) f3_t(t)$numpy(), t))
```

    ## [1] 0.5

``` r
check.derivatives(.x = t, func = function(t) f3_t(t)$numpy(), func_grad = r_grad_r.f3_t)
```

    ## Error in py_call_impl(callable, dots$args, dots$keywords): InvalidArgumentError: cannot compute Less as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:Less]
    ## 
    ## Detailed traceback: 
    ##   File "/home/asoen/anaconda3/envs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_math_ops.py", line 5376, in less
    ##     _six.raise_from(_core._status_to_exception(e.code, message), None)
    ##   File "<string>", line 3, in raise_from

So progress, but not quite there. Lets convert in the input as well now.

``` r
t_convert <- tf$convert_to_tensor(t, tf$float64)
print(t)
```

    ## [1] 2

``` r
print(t_convert)
```

    ## tf.Tensor(2.0, shape=(), dtype=float64)

``` r
print(f3_t(t_convert))
```

    ## tf.Tensor(6.830203569361039, shape=(), dtype=float64)

``` r
print(r_grad_r.f3_t(t_convert))
```

    ## [1] 0.5

``` r
print(grad(function(t) f3_t(tf$convert_to_tensor(t, tf$float64))$numpy(), t))
```

    ## [1] 0.5

``` r
check.derivatives(.x = t, func = function(t) f3_t(tf$convert_to_tensor(t, tf$float64))$numpy(), func_grad = function(t) r_grad_r.f3_t(tf$convert_to_tensor(t, tf$float64)))
```

    ## Derivative checker results: 0 error(s) detected.

    ## 
    ##   grad_f[ 1 ] = 5e-01 ~ 5.000001e-01   [1.788139e-07]

    ## $analytic
    ## [1] 0.5
    ## 
    ## $finite_difference
    ## [1] 0.5000001
    ## 
    ## $relative_error
    ## [1] 1.788139e-07
    ## 
    ## $flag_derivative_warning
    ## [1] FALSE

Its a bit of a pain, but you will need to make sure that the types work
if we want to use/compare against these numeric solvers.

#### Weird Trick for Lists

Lets just change the input to a list and just try and get the tensorflow
gradient.

``` r
ts <- c(1, 3, 2, 2, 3)

print(f3_t(ts) %>% k_sum)
```

    ## tf.Tensor(7.978285042333477, shape=(), dtype=float64)

``` r
print(r_grad_r.f3_t(ts) %>% k_sum)
```

    ## Error in py_call_impl(callable, dots$args, dots$keywords): InvalidArgumentError: cannot compute Less as input #1(zero-based) was expected to be a double tensor but is a float tensor [Op:Less]
    ## 
    ## Detailed traceback: 
    ##   File "/home/asoen/anaconda3/envs/r-reticulate/lib/python3.6/site-packages/tensorflow_core/python/ops/gen_math_ops.py", line 5376, in less
    ##     _six.raise_from(_core._status_to_exception(e.code, message), None)
    ##   File "<string>", line 3, in raise_from

Now lets try do a rather strange transformation of `ts`.

``` r
ts_convert <- tf$convert_to_tensor(ts, tf$float64)$numpy()
print(ts)
```

    ## [1] 1 3 2 2 3

``` r
print(ts_convert)
```

    ## [1] 1 3 2 2 3

``` r
print("")
```

    ## [1] ""

``` r
print(f3_t(ts_convert) %>% k_sum)
```

    ## tf.Tensor(7.978285042333477, shape=(), dtype=float64)

``` r
print(r_grad_r.f3_t(ts_convert) %>% k_sum)
```

    ## tf.Tensor(2.0833333333333335, shape=(), dtype=float64)

Weird right\! It seems like tensorflow will remember (at least for
lists) the previously converted datatype when using
`tf$convert_to_tensor`. Here are some more concrete examples with out
any gradients involved.

``` r
data <- c(1, 2, 3)

# is of type tf$float32
data %>% k_sum
```

    ## tf.Tensor(6.0, shape=(), dtype=float32)

``` r
# is of type tf$float32
tf$constant(c(1,2,3), tf$float64) %>% as.numeric() %>% k_sum
```

    ## tf.Tensor(6.0, shape=(), dtype=float32)

``` r
# is of type tf$float64
tf$constant(c(1,2,3), tf$float64)$numpy() %>% k_sum
```

    ## tf.Tensor(6.0, shape=(), dtype=float64)

This trick is kinda neat as we can now use our converted `ts_convert` as
per usual in R without having to worry about converting it to a
`tf$float64` explicitly.

### End to End Optimisation

TODO
