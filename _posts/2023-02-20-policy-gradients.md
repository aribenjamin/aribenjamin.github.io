---
title: Gradient estimation in a noisy world
date: Feb. 20, 2023
excerpt: How score-based generative models might be the secret to good reinforcement learning in noisy worlds.
usemathjax: true
---
<p style="margin-bottom:2cm;font-size:.8em;font-style:italic">
(This is part of my *steal my idea* series. If you want to pursue this project, please do! I'd like to collaborate as the project unfolds, but I'm not available to be a first author. Send me an email if you're interested and I can tell you who else may be working on this already.)
</p>

Suppose we have an agent with control over its actions. The environment responds stochastically to these actions, and each action has some associated reward. How can we maximize the expected rewards, given our action-selection policy? 

We will define the *policy* of an RL agent is its probability distribution over particular action, $$\pi_\theta(a_t\vert s_t)$$, given the current state $$s_t$$. This policy might be specified by some neural network with parameters $$\theta$$. After an action is taken, there is some state $$s_{t+1}$$ that pulls from $$P(s_{t+1}\vert  a_t)$$, and some reward assocated with that state, $$R(s_{t+1}).$$ It's important to recognize there are **two** sources of stochasticity here. The first is the stochasticity our policy. The second is the stochasticity of the world given our actions.  

The expected reward of the **next** timestep under our policy, given both sources of stochasticity, can be written as:

 $$V^\pi(s_{t+1})=\mathbb{E}_{\pi_\theta(a_t\vert s_t)}\left[\mathbb{E}_{P(s_{t+1}\vert  a_t)}[R(s_{t+1})]\right]$$. 
 
 The first expectation is due to stochasticity in the policy, and the second is due to stochasticity in the world. For simplicity let's ignore the far future and only look one step ahead. 

## Estimating the policy gradient

In order to improve this expected reward, we need to follow its gradient with respect to the parameters of the policy, $$\nabla_{\theta}V^\pi.$$

The first step here is to get gradients of a stochastic policy using the **reparameterization trick**. At this point this is similar to the Deterministic Policy Gradient (DPG) approach.  

For reparameterization we have to be assume that the policy is a deterministic, differentiable function of a different random variable $$\epsilon$$. Then we can pull the gradient inside of the expectation (because now the expectation is only over the stochastic variable). Letting $$a_t = \alpha(\epsilon,\theta,s_t)$$, for some function $$\alpha$$, we have:

$$\nabla_{\theta}\mathbb{E}_{\pi_\theta(a_t\vert s_t)}\left[h(a_t)\right]=\mathbb{E}_\epsilon [\nabla_{\theta}h(\alpha(\epsilon,\theta,s_t))]$$

This is a powerful tool. Above I've put the function we're maximizing as just some stand-in $$h(a_t)$$. We can go further with the chain rule:

$$\nabla_{\theta}\mathbb{E}_{\pi_\theta(a_t\vert s_t)}\left[h(a_t)\right]=\mathbb{E}_\epsilon [\nabla_{a_t}h(a_t)\cdot\nabla_\theta \alpha(\epsilon,\theta,s_t)]$$

Thus, with reparameterization, we can split our desired gradient into two problems using the chain rule:
1. First, what is the gradient of our value with respect to our policy, $$\nabla_a \mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_{t+1})]$$? 
2. Second, what is the gradient of our policy with respect to our parameters, $$\nabla_\theta a$$? 

This separation into two problems mirrors the two sources of stochasticity here (policy and environmental).  

There's just one problem here: we can't backpropagate through our environment! The above stand-in $$h(a_t)$$ is the expectation $$\mathbb{E}_{P(s_{t+1}\vert  a_t)}[R(s_{t+1})]$$, and we can't take the gradient of this w/r/t $$a_t$$. What should we do? One option is to ignore the environmental stochasticity. This might work well for games. However, this will not work for the real world.  

To solve this problem we will need some other way of estimating $$\nabla_{a_t}\mathbb{E}_{P(s_{t+1}\vert  a_t)}[R(s_{t+1})]$$.

## Estimating the gradient with respect to $$a_t$$

To estimate $$\nabla_{a_t} \mathbb{E}_{P(s_{t+1}\vert  a_t)}[R(s_{t+1})]$$ we'll do two things. First, we'll use the REINFORCE trick, and then we'll plug in for some score estimators.

Written generally, the REINFORCE trick is the identity,

$$\nabla_{\theta}\mathbb{E}_{p(x;\theta)}[f(x)] = \mathbb{E}_{p(x;\theta)}[\nabla_{\theta} \left(\log p(x;\theta)\right) f(x) ]. $$

Applied to our situation, this means that,

$$\nabla_{a_t} \mathbb{E}_{p(s_{t+1}\vert  a_t)}[R(s_{t+1})]=\mathbb{E}_{p(s_{t+1}\vert  a_t)}[R(s_t) \nabla_{a_t} \log p(s_{t+1}\vert  a_t)] .$$  

Unfortunately, we don't have access to $$\nabla_{a_t} \log p(s_{t+1}\vert  a_t)$$. We still can't backpropagate through the world!  

Though we don't have access to $$\nabla_{a_t} \log p(s_{t+1}\vert  a_t)$$, we can estimate it. Could we, say, train a neural network to predict $$s_{t+1}$$ given $$a_t$$, and then follow the gradient with respect to $$a_t$$? My guess is no.

#### Why we can't train a discriminative classifer for $$\nabla_{a} \log p(s_{t+1}\vert  a)$$ : adversarial fragility.

Now, one thing we could do is to train up an approximate neural network to approximate $$\log p(s_{t+1}\vert  a_t)$$, and then use backprop to calculate $$\nabla_{a_t}\log p(s_{t+1}\vert  a_t)$$. This is unlikely to be successful, because these will be adversarial examples. The gradient of a neural network classifer is not a robust estimator of the gradient of the (true) input/output relation.

#### An alternative strategy: use Bayes' Rule

We could also take a generative modeling approach. Using Bayes' rule, we can flip around the problematic $$\nabla_{a_t} \log p(s_{t+1}\vert  a_t)$$ into something else with better properties. First, with Bayes Rule,

$$ \nabla_{a_t} \log p(s_{t+1}\vert  a_t) = \nabla_{a_t} \left(\log p(a_t\vert  s_{t+1}) + \log p(s_{t+1}) - \log p(a_t)\right) $$

Taking the derivative, the $$p(s_{t+1})$$ goes away as it does not depend on $$a$$:

$$\nabla_{a_t} \log p(s_{t+1}\vert  a_t)= \nabla_{a_t} \log p(a_t\vert  s_{t+1}) - \nabla_{a_t} \log p(a_t) $$

Thus, the derivative we need is the difference between two score functions over our policy: one conditioned on the outcome, and the other unconditioned. Perhaps 10 years ago this would have seemed like a dead end. But now there are wonderful methods for estimating such score functions.

### Estimating the policy score with denoising autoencoders (a.k.a. the diffusion approach)

A beautiful result from recent generative modeling literature is that the score of a probability density can easily be obtained with a good denoiser. Specifically, $$\nabla_{a_t} \log p(a_t) \approx \hat{a_t}-a_t$$, where $$\hat{a_t}$$ is the output of a neural network attempting to denoise $$a_t$$ from Gaussian noise of unit variance.

I particularly like the proof of Kadkhodaie & Simoncelli, which I'll reproduce below as a footnote [^2]. This is the magic of diffusion models, which underlie Midjourney, Stable Diffusion, etc. Intuitively, these work by "denoising noise". Because the denoising estimate is an estimate of the score, these generate images by walking up a learned score function towards more likely images.

If we obtain two denoising estimates of $$\hat{a_t}$$, one unconditional and the other conditioned on the outcome $$s_{t+1}$$, we can perform this very simple subtraction to get an estimate of the gradient:

$$\nabla_{a_t} \log p(_{t+1}\vert  a_t)= (\hat{a_t}_{\vert  s_{t+1}}-a_t) - (\hat{a_t}-a_t) =  \hat{a_t}_{\vert  s_{t+1}} - \hat{a_t}.$$

This is remarkably simple. Looking to the strengths of diffusion models, furthermore, we can expect this estimator will have these advantages:
1. It will work well in high-dimensional action spaces.
2. It will benefit from significant pretraining in sessions without external reward.
3. It will be much more robust, in the adversarial sense.

To convince you of the third point, I decided to create a "canonical 9" using this roundabout method. I trained a denoising diffusion model on MNIST, using one network for both unconditional and conditional denoising[^4]. Using this pretrained denoising diffusion model, we can walk the pixels following $$\nabla_x\log p(y=9\vert  x)$$ via the difference of denoising estimates $$\hat{x}_{9} - \hat{x}$$. 

<figure>
  <img src="{{site.baseurl}}/assets/images/gradient_estimation/9_.gif" data-action="zoom" style="width:300px;" class="centerImage" >
</figure>

### Putting it together: Denoising policy gradients for noisy environments

Here's our expression after plugging the denoising score estimators in:

$$\nabla_{a_t} \log p(s_{t+1}\vert  a_t) =\mathbb{E}_{p(s_{t+1}\vert  a_t)}[R(s_{t+1}) (\hat{a_t}_{\vert  s_{t+1}} - \hat{a_t})  ]$$

This includes two neural networks or function approximators:
1. The action network, $$a_t = \alpha(s,\epsilon, \theta)$$.
2. The (denoising) action prediction network, which can optionally be conditioned on the outcome, $$\hat{a}_t = f(a_t, s_t, s\in\{s_{t+1},\varnothing\})$$.

Training the action network requires a good action prediction network. The action prediction network can be trained with and without conditioning on $$s_{t+1}$$ on a Gaussian denoising objective.

### Extension: control variates

To reduce the variance of this estimator, we can include a reward prediction network or any of the standard tricks of RL. We are dealing with environmental noise with the REINFORCE method, and any methods that work there will port over here.

If we train a reward prediction $$\hat{R}(s_{t+1})=b(s_t,a_t)$$ as a baseline, we can plug this in to reduce the variance while remaining unbiased. The gradient with respect to our action can then be written as:

$$\nabla_{a_t} \log p(s_{t+1}\vert  a_t) =\mathbb{E}_{p(s_{t+1}\vert  a_t)}[(R(s_{t+1})-b(s_t,a_t)) (\hat{a_t}_{\vert  s_{t+1}} - \hat{a_t})]$$

## Putting it together: Denoising policy gradients for noisy environments

Finally, we have an expression for $$\nabla_{a_t} \log p(s_{t+1}\vert  a_t)$$ that we can use with a reparameterized policy network.

First, after the reparameterization trick but before invoking the denoising gradient approximation, we had 

$$\nabla_{\theta}V^\pi =\mathbb{E}_\epsilon \left[\nabla_\theta \left[\mathbb{E}_{p(s_{t+1}\vert  a_t)}[R(s_{t+1})) ]\right]\cdot\nabla_\theta \alpha(\epsilon,\theta,s_t)\right]$$

Then, after REINFORCE, we had,

$$\nabla_{\theta}V^\pi =\mathbb{E}_\epsilon \left[\mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_t) \nabla_{a_t} \log p(s_{t+1}\vert  a_t)\cdot \nabla_\theta \alpha(s_t,\epsilon,\theta) ]\right]$$

And then finally with the denoising score estimators:

$$\nabla_{\theta}V^\pi =\mathbb{E}_\epsilon \left[\mathbb{E}_{p(s_{t+1}\vert  a_t)}[((R(s_{t+1})-b(s_t,a_t)) (\hat{a_t}_{\vert  s_{t+1}} - \hat{a_t}))\cdot \nabla_\theta \alpha(s_t,\epsilon,\theta) ]\right].$$

And that's it! We succeeded at reparameterizing our policy network while allowing for a noisy world.

To review, this involves three networks,
1. The action network, $$a_t=\alpha(s_t,\epsilon, \theta)$$.
2. The (denoising) action prediction network, which can optionally be conditioned on reward, $$\hat{a} = f(a_t, s_t,s\in\{s_{t+1},\varnothing\})$$, trained with and without conditioning on $$s_{t+1}$$ on a denoising objective.
3. The reward prediction network, $$b(s_t,a_t)$$, trained as normal. 

## Next steps

Cool, cool -- but does this all work? The next step would be to extend this to longer time horizons than one step ahead. I haven't worked this out yet, but it should just be a matter of using the standard Bellman equations.

___
*Footnotes*

[^1]: Schulman, J., Heess, N., Weber, T., & Abbeel, P. (2015). Gradient estimation using stochastic computation graphs. Advances in neural information processing systems, 28.

[^2]: From Kadkhodaie, Z., & Simoncelli, E. P. (2020). Solving linear inverse problems using the prior implicit in a denoiser. arXiv preprint arXiv:2007.13640. Here's the derivation. First, we noise an action $$a$$ with Gaussian noise: $$\tilde{a} = a + \eta$$, $$\eta\sim\mathcal{N}(0,\sigma^2)$$. The joint distribution of noised and unnoised $$a$$ we'll write as $$p(\tilde{a},a).$$ Then, the gradient of the noised distribution is a convolution of the Gaussian envelope over the original data. 1)  $$\nabla_{\tilde{a}}p(\tilde{a})=\frac1{\sigma^2}\int(a-\tilde{a})\mathcal{N}(a; 0,\sigma^2)p(a)da$$ 2) $$=\frac1{\sigma^2}\int(a-\tilde{a})p(\tilde{a},a)da$$  Dividing by $$p(\tilde{a})$$ on both sides and splitting the integral,  $$\frac{\nabla_{\tilde{a}}p(\tilde{a})}{p(\tilde{a})}\sigma^2=\int a p(a\vert  \tilde{a}) da -\int \tilde{a} p(a\vert  \tilde{a})da.$$  The left-hand side simplifies to a derivative of a log. On the right, the left term is the mean of the posterior over $$a$$, which minimizes the MSE. Thus it is the optimal denoising estimate $$\hat{a}$$ given an observation of a noisy $$\tilde{a}$$. The right simplifies to $$y$$. Then,  $$\nabla_{\tilde{a}}\log p(\tilde{a})\sigma^2=\hat{a}-\tilde{a}.$$  Note this is the score of the noised distribution. For small enough $\sigma$, this will be quite close to score of the original.<br> 

[^3]: https://github.com/lucidrains/denoising-diffusion-pytorch