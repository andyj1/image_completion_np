NeuralProcess(
  (encoder): Encoder(
    (layers): Sequential(
      (0): Sequential(
        (0): Linear(in_features=3, out_features=400, bias=True)
        (1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Linear(in_features=400, out_features=400, bias=True)
        (1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (2): Linear(in_features=400, out_features=256, bias=True)
    )
  )
  (latent_encoder): LatentEncoder(
    (mean_aggregater): _Aggregator(
      (layers): Sequential(
        (0): Linear(in_features=256, out_features=400, bias=True)
        (1): LeakyReLU(negative_slope=0.01)
        (2): Linear(in_features=400, out_features=400, bias=True)
      )
    )
    (variance_aggregator): _Aggregator(
      (layers): Sequential(
        (0): Linear(in_features=256, out_features=400, bias=True)
        (1): LeakyReLU(negative_slope=0.01)
        (2): Linear(in_features=400, out_features=400, bias=True)
      )
    )
  )
  (decoder): Decoder(
    (xr_to_hidden): Sequential(
      (0): Sequential(
        (0): Linear(in_features=402, out_features=400, bias=True)
        (1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (1): Sequential(
        (0): Linear(in_features=400, out_features=400, bias=True)
        (1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (2): Sequential(
        (0): Linear(in_features=400, out_features=400, bias=True)
        (1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (3): Sequential(
        (0): Linear(in_features=400, out_features=400, bias=True)
        (1): BatchNorm1d(400, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU(inplace=True)
      )
      (4): Linear(in_features=400, out_features=2, bias=True)
    )
  )
)
