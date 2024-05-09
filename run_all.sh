
# MLP
#   MNIST - class shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_class --dataset=MNIST --top_dir=shuffle_class_mnist_mlp --device=cuda:1 > shuffle_class_mnist_mlp.txt 2>&1 &
#   MNIST  - full shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_full --dataset=MNIST --top_dir=shuffle_full_mnist_mlp --device=cuda:1 > shuffle_full_mnist_mlp.txt 2>&1 &
#   CIFAR10 - class shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_class --dataset=CIFAR10 --top_dir=shuffle_class_cifar10_mlp --device=cuda:2 --learning_rate=0.00005 > shuffle_class_cifar10_mlp.txt 2>&1 &
#   CIFAR10 - full shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_full --dataset=CIFAR10 --top_dir=shuffle_full_cifar10_mlp --device=cuda:2 --learning_rate=0.00005 > shuffle_full_cifar10_mlp.txt 2>&1 &

# CNN
#   MNIST - class shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_class --dataset=MNIST --top_dir=shuffle_class_mnist_cnn --model=CNN --learning_rate=0.01 --device=cuda:2 > shuffle_class_mnist_cnn.txt 2>&1 &
#   MNIST - full shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_full --dataset=MNIST --top_dir=shuffle_full_mnist_cnn --model=CNN --learning_rate=0.01 --device=cuda:2 > shuffle_full_mnist_cnn.txt 2>&1 &
#   CIFAR10 - class shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_class --dataset=CIFAR10 --top_dir=shuffle_class_cifar10_cnn --model=CNN --learning_rate=0.001 --device=cuda:1 > shuffle_class_cifar10_cnn.txt 2>&1 &
#   CIFAR10 - full shuffle
python -u project.py --epochs=10 --restarts=30 --new_labels=shuffle_full --dataset=CIFAR10 --top_dir=shuffle_full_cifar10_cnn --learning_rate=0.001 --model=CNN --device=cuda:1 > shuffle_full_cifar10_cnn.txt 2>&1 &
