def get_parameter_groups(model, cfg):
    return [
        {
            "params": [parameter for parameter in model.parameters() if parameter.requires_grad],
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        }
    ]
