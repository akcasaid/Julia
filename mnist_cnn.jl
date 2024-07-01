using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, crossentropy, throttle
using Base.Iterators: repeated
using CUDA

# Veri setini yükle
imgs = MNIST.images()
labels = MNIST.labels()

# Verileri ön işleme
X = Float32.(reshape.(imgs, :)) |> gpu
Y = onehotbatch(labels, 0:9) |> gpu

# Eğitim ve test setlerini ayır
(X_train, X_test) = splitobs(X, at=0.8)
(Y_train, Y_test) = splitobs(Y, at=0.8)


model = Chain(
  Conv((3, 3), 1=>16, relu),
  MaxPool((2,2)),
  Conv((3, 3), 16=>32, relu),
  MaxPool((2,2)),
  Conv((3, 3), 32=>64, relu),
  flatten,
  Dense(576, 128, relu),
  Dense(128, 10),
  softmax
) |> gpu

# Kayıp fonksiyonu
loss(x, y) = crossentropy(model(x), y)

# Optimizatör
optimizer = ADAM()

# Eğitim döngüsü
dataset = repeated((X_train, Y_train), 200)
evalcb = () -> @show(loss(X_test, Y_test))

Flux.train!(loss, params(model), dataset, optimizer, cb = throttle(evalcb, 10))
