---
title: Gradient estimation in a noisy world
date: Feb. 20, 2023
excerpt: How score-based generative models might be the secret to good reinforcement learning in noisy worlds.
usemathjax: true
---
<!-- <p style="margin-bottom:2cm;font-size:.8em;font-style:italic">
(This is part of my *steal my idea* series. I often have more ideas than time to work on them. If you want to pursue this project, please do! I'd like to be an author and  collaborate as the project unfolds, but I'm not available to be a first author. Send me an email if you're interested and I can tell you who else may be working on this already.)
</p> -->

A common problem in reinforcement learning and in theoretical neuroscience is **gradient estimation**. We have control over our actions (to some degree), and want to optimize them. Observing a reward, we want to update our actions to increase the likelihood of that reward occuring. 

In this post I go back to the basics and derive a new gradient estimator designed for use in noisy environments. I think it should work well in large action spaces, and deals naturally with the problems of *noisy worlds*, *adversarial fragility*, and *sparse rewards*. 

## Self-improving in a stochastic world.
<!-- <figure>
  <img src="{{site.baseurl}}/assets/images/gradient_estimation/schema.png" data-action="zoom" style="width:400px;" class="centerImage">
</figure> -->

Suppose we have an agent with control over its actions. The environment responds stochastically to these actions, and each action has some associated reward. How can we maximize the expected rewards, given our action-selection policy?

This is the classic setup of policy gradients. The *policy* of an RL agent is its probability distribution over particular action, $$p(a)$$. This policy may be a lookup table listing the probabilities over all possible $$a$$, or it might be specified by some neural network with parameters $$\theta$$, as $$a=f(s_t, \theta)$$. After an action is taken, there is some state $$s_{t+1}$$ that pulls from $$p(s_{t+1}\vert  a)$$, and some reward assocated with that state, $$R(s_{t+1}).$$

It's important to recognize there are **two** sources of stochasticity here. The first is the stochasticity our policy. The second is the stochasticity of the world given our actions.

In this post I am primarily interested in this second source of stochasticity – the stochasticity of the world, given our actions. Let's write this down. Reality is noisy, and there is a probabilistic relation between $$a$$ and $$s_{t+1}$$, given conditionally as $$p(s_{t+1}\vert  a)$$. The expected reward is then 
$$V^\pi=\mathbb{E}_{p(a;\theta)}\left[\mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_{t+1})]\right]$$. The first expectation is due to stochasticity in the policy, and the second is due to stochasticity in the world. 

In order to improve this expected reward, we need to follow its gradient with respect to the parameters of the policy, $$\nabla_{\theta}V^\pi$$. We can split our desired gradient into two problems using the chain rule:
1. First, what is the gradient of our value with respect to our policy, $$\nabla_a \mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_{t+1})]$$? 
2. Second, what is the gradient of our policy with respect to our parameters, $$\nabla_\theta a$$? 

This separation into two problems mirrors the two sources of stochasticity here (policy and environmental). 

## Step 1: Estimating the value gradient, with respect to the policy

To estimate $$\nabla_a \mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_{t+1})]$$ we'll use the REINFORCE trick. You can see this footnote[^1] for a derivation; this trick is the identity,

$$\nabla_{\theta}\mathbb{E}_{p(x;\theta)}[f(x)] = \mathbb{E}_{p(x;\theta)}[\nabla_{\theta} \left(\log p(x;\theta)\right) f(x) ]. $$

Applied to our situation, this means that,

$$\nabla_a \mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_{t+1})]=\mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_t) \nabla_{a} \log p(s_{t+1}\vert  a)] .$$

Unfortunately, we don't have access to $$\nabla_{a} \log p(s_{t+1}\vert  a)$$. We can't backpropagate through the world! Without this, we won't be able to account for the stochasticity of reward, given our policy. I think this is a major downside of current RL. As far as I can tell, the usual approach is to just sweep the environmental stochasticity into the policy stochasticity and pretend it doesn't exist.

Though we don't have access to $$\nabla_{a} \log p(s_{t+1}\vert  a)$$, we can estimate it. Could we, say, train a neural network to predict $$s_{t+1}$$ given $$a$$, and then follow the gradient with respect to $$a$$? My guess is no.

#### Why we can't train a discriminative classifer for $$\nabla_{a} \log p(s_{t+1}\vert  a)$$ : adversarial fragility.

Now, one thing we could do is to train up an approximate neural network to approximate $$\log p(s_{t+1}\vert  a)$$, and then use backprop to calculate $$\nabla_{a}\log p(s_{t+1}\vert  a)$$. This is unlikely to be successful, because these will be adversarial examples. The gradient of a neural network classifer is not a robust estimator of the gradient of the (true) input/output relation.

#### An alternative strategy: use Bayes' Rule

We could also take a generative approach. Using Bayes' rule, we can flip around the problematic $$\nabla_{a} \log p(s_{t+1}\vert  a)$$ into something else with better properties. First, with Bayes Rule,

$$ \nabla_{a} \log p(s_{t+1}\vert  a) = \nabla_{a} \left(\log p(a\vert  s_{t+1}) + \log p(s_{t+1}) - \log p(a)\right) $$

Taking the derivative, the $$p(s_{t+1})$$ goes away as it does not depend on $$a$$:

$$\nabla_{a} \log p(s_{t+1}\vert  a)= \nabla_{a} \log p(a\vert  s_{t+1}) - \nabla_{a} \log p(a) $$

Thus, the derivative we need is the difference between two score functions over our policy: one conditioned on the outcome, and the other unconditioned. Perhaps 10 years ago this would have seemed like a dead end. But now there are wonderful methods for estimating such score functions.

### Estimating the policy score with denoising autoencoders (a.k.a. the diffusion approach)

A beautiful result from recent generative modeling literature is that the score of a probability density can easily be obtained with a good denoiser. Specifically,
$$\nabla_{a} \log p(a) \approx \hat{a}-a,$$ where $$\hat{a}$$ is the output of a neural network attempting to denoise $$a$$ from Gaussian noise of unit variance.

I particularly like the proof of Kadkhodaie & Simoncelli, which I'll reproduce below as a footnote [^2]. This is the magic of diffusion models, which underlie Midjourney, Stable Diffusion, etc. Intuitively, these work by "denoising noise". Because the denoising estimate is an estimate of the score, these generate images by walking up a learned score function towards more likely images.

If we obtain two denoising estimates of $$\hat{a}$$, one unconditional and the other conditioned on the outcome $$s_{t+1}$$, we can perform this very simple subtraction to get an estimate of the gradient:

$$\nabla_{a} \log p(_{t+1}\vert  a)= (\hat{a}_{\vert  s_{t+1}}-a) - (\hat{a}-a) =  \hat{a}_{\vert  s_{t+1}} - \hat{a}.$$

This is remarkably simple. Looking to the strengths of diffusion models, furthermore, we can expect this estimator will have these advantages:
1. It will work well in high-dimensional action spaces.
2. It will benefit from significant pretraining in sessions without external reward.
3. It will be much more robust, in the adversarial sense.

To convince you of the third point, I decided to create a "canonical 9" using this roundabout method. I trained a denoising diffusion model on MNIST, using one network for both unconditional and conditional denoising[^4]. Using this pretrained denoising diffusion model, we can walk the pixels following $$\nabla_x\log p(y=9\vert  x)$$ via the difference of denoising estimates $$\hat{x}_{9} - \hat{x}$$. 

<figure>
  <img src="{{site.baseurl}}/assets/images/gradient_estimation/9_.gif" data-action="zoom" style="width:300px;" class="centerImage" >
</figure>

### Putting it together: Denoising policy gradients for noisy environments

Here's our final expression after plugging things in:

$$\nabla_{a} \log p(s_{t+1}\vert  a) =\mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_{t+1}) (\hat{a}_{\vert  s_{t+1}} - \hat{a})  ]$$

This includes two neural networks or function approximators:
1. The action network, $$a = f(s,\epsilon, \theta)$$.
2. The (denoising) action prediction network, which can optionally be conditioned on the outcome, $$\hat{a} = g(a, r\in\{s_{i+1},s_i})$$.

Training the action network requires a good action prediction network and would be done with the standard Bellman equations. The action prediction network can be trained with and without conditioning on $$s_{t+1}$$ on a Gaussian denoising objective.

### Extension: control variates

To reduce the variance of this estimator, we can include a reward prediction network or any of the standard tricks of RL. We are dealing with environmental noise with the REINFORCE method, and any methods that work there will port over here.

If we train a reward prediction $$\hat{R(s_{t+1})}=b(s_i,a)$$ as a baseline, we can plug this in to reduce the variance while remaining unbiased. The  policy gradient can then be written as:

$$\nabla_{a} \log p(s_{t+1}\vert  a) =\mathbb{E}_{p(s_{t+1}\vert  a)}[(R(s_{t+1})-b(s_i,a)) (\hat{a}_{\vert  s_{t+1}} - \hat{a})]$$


## Finally: the gradient w/r/t parameters 

Now that we have an expression for the gradient with respect to our policy, we can use that to backpropagate to our policy's parameters.

To promote exploration, we likely want our policy to be stochastic as well. (Remember, this was our first source of stochasticity). To calculate the gradient through this stochastic policy, we could either 1) use the REINFORCE trick again, or 2) use the **reparameterization trick**. (I don't know why these are undersold as 'tricks'; they're both quite cool. Stop underselling your work, people!) As a rule, if you can control your stochasticity, reparameterize it. Otherwise, you're stuck with REINFORCE.

For **reparameterization** we have to be assume that the policy is a deterministic, differentiable function of a different random variable $$\epsilon$$. Then we can pull the gradient inside of the expectation (because now the expectation is only over the stochastic variable). Letting $$x = g(\epsilon,\theta)$$, we have:

 $$\nabla_{\theta}\mathbb{E}_{p(x;\theta)}[f(x)] =\mathbb{E}_\epsilon [\nabla_{\theta}f(g(\epsilon,\theta))]$$

This is called the reparameterization trick, or the pathwise derivative. 

We'd love to just use reparameterization, but unfortunately we can't reparameterize the noise in our environment, hence this project. To fully use this approach, we could assume that rewards are deterministic, which will work for games but not the real world.

### Reparameterize your policy; REINFORCE for the world

Let's *combine* approaches #1 and #2[^3] and reparameterize our policy as $$a=f(s_t,\epsilon,\theta)$$. For our two sources of noise we are using two different styles of gradient estimation – reparameterization for the policy noise, and REINFORCE for the world noise. First, before invoking the denoising gradient approximation, we have

$$\nabla_{\theta}V^\pi =\mathbb{E}_\epsilon \left[\nabla_\theta \left[\mathbb{E}_{p(s_{t+1}\vert  a = f(s_t,\epsilon,\theta))}[R(s_{t+1})) ]\right]\right]$$

After the chain rule (remembering that $$\theta\rightarrow a\tilde{\rightarrow}s_{i+1}$$), we have:

$$\nabla_{\theta}V^\pi =\mathbb{E}_\epsilon \left[\mathbb{E}_{p(s_{t+1}\vert  a)}[R(s_t) \nabla_{a} \log p(s_{t+1}\vert  a)^T \nabla_\theta f(s_t,\epsilon,\theta) ]\right]$$


### Putting it together: Denoising policy gradients for noisy environments

Here's our final expression after plugging things in:

$$\nabla_{\theta}V^\pi =\mathbb{E}_\epsilon \left[\mathbb{E}_{p(s_{t+1}\vert  a)}[((R(s_{t+1})-b(s_i,a)) (\hat{a}_{\vert  s_{t+1}} - \hat{a}))^T \nabla_\theta f(s_t,\epsilon,\theta) ]\right].$$

And that's it! 

To review, this involves three networks,
1. The action network, $$a = f(s,\epsilon, \theta)$$, trained with the standard Bellman equations. 
2. The (denoising) action prediction network, which can optionally be conditioned on reward, $$\hat{a} = g(a, s_{i+1}\in\{s_{i+1},\varnothing\})$$, trained with and without conditioning on $$s_{t+1}$$ on a denoising objective.
3. The reward prediction network, $$b(s_i,a)$$, trained as normal. 

___
*Footnotes*

[^1]: Here's the derivation of REINFORCE.

  $$\nabla_{\theta}\mathbb{E}_{p(a;\theta)}[R(a)] = \nabla_{\theta}\int_a p(a;\theta) R(a)$$

  $$=  \int_a  R(a) \nabla_{p(a;\theta)} p(a;\theta) $$

  $$=  \int_a  R(a) p(a) \nabla_{p(a;\theta)} \log p(a;\theta)$$

  $$=\nabla_{\theta}\mathbb{E}_{p(a;\theta)}[\nabla_{p(a;\theta)} \log p(a;\theta)  R(a) ] $$

[^3]: Schulman, J., Heess, N., Weber, T., & Abbeel, P. (2015). Gradient estimation using stochastic computation graphs. Advances in neural information processing systems, 28.

[^2]: From Kadkhodaie, Z., & Simoncelli, E. P. (2020). Solving linear inverse problems using the prior implicit in a denoiser. arXiv preprint arXiv:2007.13640.

  Here's the derivation. First, we noise an action $$a$$ with Gaussian noise: $$\tilde{a} = a + \eta$$, $$\eta\sim\mathcal{N}(0,\sigma^2)$$. The joint distribution of noised and unnoised $$a$$ we'll write as $$p(\tilde{a},a).$$ Then, the gradient of the noised distribution is a convolution of the Gaussian envelope over the original data.

  $$\nabla_{\tilde{a}}p(\tilde{a})=\frac1{\sigma^2}\int(a-\tilde{a})\mathcal{N}(a; 0,\sigma^2)p(a)da$$

  $$=\frac1{\sigma^2}\int(a-\tilde{a})p(\tilde{a},a)da$$

  Dividing by $$p(\tilde{a})$$ on both sides and splitting the integral,

  $$\frac{\nabla_{\tilde{a}}p(\tilde{a})}{p(\tilde{a})}\sigma^2=\int a p(a\vert  \tilde{a}) da -\int \tilde{a} p(a\vert  \tilde{a})da$$

  The left-hand side simplifies to a derivative of a log. On the right, the left term is the mean of the posterior over $$a$$, which minimizes the MSE. Thus it is the optimal denoising estimate $$\hat{a}$$ given an observation of a noisy $$\tilde{a}$$. The right simplifies to $$y$$. 

  $$\nabla_{\tilde{a}}\log p(\tilde{a})\sigma^2=\hat{a}-\tilde{a}$$

  Note this is the score of the noised distribution. For small enough $\sigma$, this will be quite close to score of the original.<br> One interesting aspect of this derivation is that we only invoked the properties of Gaussian noise once: in noting that $\int a p(a\vert  \tilde{a}) da $ is the mean-squared error estimator of the unnoised input. This is true of other noise distributions, as well; the noise need not be Gaussian.

[^4]: https://github.com/lucidrains/denoising-diffusion-pytorch