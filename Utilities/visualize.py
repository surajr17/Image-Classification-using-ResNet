import matplotlib.pyplot as plt
from torchvision import transforms

def plot_class_label_counts(data_loader, classes):
    class_counts = {}
    for class_name in classes:
        class_counts[class_name] = 0
    for _, batch_labels in data_loader:
        for label in batch_labels:
            class_counts[classes[label.item()]] += 1

    fig = plt.figure()
    plt.suptitle("Class Distribution")
    plt.bar(range(len(class_counts)), list(class_counts.values()))
    plt.xticks(range(len(class_counts)), list(class_counts.keys()), rotation=90)
    plt.tight_layout()
    plt.show()

def plot_data_samples(data_loader, classes):
    batch_data, batch_label = next(iter(data_loader))

    fig = plt.figure()
    plt.suptitle("Data Samples with Labels post Transforms")
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        # unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(batch_data[i])
        plt.imshow(transforms.ToPILImage()(unnormalized))
        plt.title(
            classes[batch_label[i].item()],
        )

        plt.xticks([])
        plt.yticks([])

def plot_model_training_curves(train_accs, test_accs, train_losses, test_losses):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_accs)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_accs)
    axs[1, 1].set_title("Test Accuracy")
    plt.plot()

def plot_incorrect_preds(incorrect, classes, num_imgs):
    # num_imgs is a multiple of 5
    assert num_imgs % 5 == 0
    assert len(incorrect) >= num_imgs

    # incorrect (data, target, pred, output)
    print(f"Total Incorrect Predictions {len(incorrect)}")
    fig = plt.figure(figsize=(10, num_imgs // 2))
    plt.suptitle("Target | Predicted Label")
    for i in range(num_imgs):
        plt.subplot(num_imgs // 5, 5, i + 1, aspect="auto")

        # unnormalize = T.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
        unnormalized = transforms.Normalize(
            (-1.98947368, -1.98436214, -1.71072797), (4.048583, 4.11522634, 3.83141762)
        )(incorrect[i][0])
        plt.imshow(transforms.ToPILImage()(unnormalized))
        plt.title(
            f"{classes[incorrect[i][1].item()]}|{classes[incorrect[i][2].item()]}",
            # fontsize=8,
        )
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()