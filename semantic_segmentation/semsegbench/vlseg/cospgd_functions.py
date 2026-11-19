"""Projection steps from cospgd 0.1.0.

``from cospgd import functions`` is an ``Attack`` instance. ``EncoderDecoder``
calls these static methods. The source is the MIT-licensed file
``cospgd/attack_implementations.py`` at
https://github.com/shashankskagnihotri/cospgd
(``__init__.py`` sets ``functions = Attack()``).
"""

import torch


class Attack:
    @staticmethod
    def step_inf(perturbed_image, epsilon, data_grad, orig_image, alpha, targeted, clamp_min=0, clamp_max=1, grad_scale=None):
        sign_data_grad = alpha * data_grad.sign()
        if targeted:
            sign_data_grad *= -1
        if grad_scale is not None:
            sign_data_grad *= grad_scale
        perturbed_image = perturbed_image.detach() + sign_data_grad
        delta = torch.clamp(perturbed_image - orig_image, min=-epsilon, max=epsilon)
        return torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()

    @staticmethod
    def step_l2(perturbed_image, epsilon, data_grad, orig_image, alpha, targeted, clamp_min=0, clamp_max=1, grad_scale=None):
        if targeted:
            data_grad *= -1
        data_grad = Attack.lp_normalize(data_grad, p=2, epsilon=1.0, decrease_only=False)
        if grad_scale is not None:
            data_grad *= grad_scale
        perturbed_image = perturbed_image.detach() + alpha * data_grad
        delta = Attack.lp_normalize(noise=perturbed_image - orig_image, p=2, epsilon=epsilon, decrease_only=True)
        return torch.clamp(orig_image + delta, clamp_min, clamp_max).detach()

    @staticmethod
    def lp_normalize(noise, p, epsilon=None, decrease_only=False):
        if epsilon is None:
            epsilon = torch.tensor(1.0)
        denom = torch.norm(noise, p=p, dim=(-1, -2, -3))
        denom = torch.maximum(denom, torch.tensor(1e-12, device=denom.device)).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        if decrease_only:
            denom = torch.maximum(denom / epsilon, torch.tensor(1.0, device=denom.device))
        else:
            denom = denom / epsilon
        return noise / denom

    @staticmethod
    def init_linf(images, epsilon, clamp_min=0, clamp_max=1):
        noise = torch.FloatTensor(images.shape).uniform_(-epsilon, epsilon).to(images.device)
        return (images + noise).clamp(clamp_min, clamp_max)

    @staticmethod
    def init_l2(images, epsilon, clamp_min=0, clamp_max=1):
        noise = torch.FloatTensor(images.shape).uniform_(-1, 1).to(images.device)
        noise = Attack.lp_normalize(noise=noise, p=2, epsilon=epsilon, decrease_only=False)
        return (images + noise).clamp(clamp_min, clamp_max)


functions = Attack()
