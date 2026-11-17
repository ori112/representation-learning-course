"""Representation evaluation: feature extraction and weighted cosine k-NN.

Piece 1 of the evaluation module. Linear probe and the collapse diagnostics
(alignment, uniformity, std, effective rank) are added in later pieces.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Run the frozen encoder over a dataloader, return (features, labels) on CPU.

    Works with or without the Trainer: the datamodule hook converts uint8 to float
    only inside a Trainer run, so the conversion is repeated here when needed.
    """
    was_training = encoder.training
    encoder.eval()  # no dropout / batch-norm updates while measuring a representation
    encoder.to(device)

    all_features, all_labels = [], []
    for images, labels in dataloader:
        images = images.to(device)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        all_features.append(encoder(images).cpu())
        all_labels.append(labels)

    encoder.train(was_training)
    return torch.cat(all_features), torch.cat(all_labels)


@torch.no_grad()
def knn_accuracy(bank_features, bank_labels, query_features, query_labels,
                 k=20, temperature=0.07, num_classes=200, chunk_size=1000):
    """Weighted cosine k-NN. Returns {"top1": ..., "top5": ...} in [0, 1].

    Each of the k nearest bank images votes for its class with weight
    exp(cosine / temperature), so close neighbors count much more than far ones.
    Raw cosine weights would let negative similarities cancel votes. Top-5 uses the
    five classes with the highest vote totals. No training is involved.
    """
    # After l2-normalisation, a dot product is the cosine similarity.
    bank = F.normalize(bank_features.float(), dim=1)
    queries = F.normalize(query_features.float(), dim=1)

    correct_top1, correct_top5 = 0, 0
    for start in range(0, len(queries), chunk_size):
        chunk = queries[start:start + chunk_size]
        similarity = chunk @ bank.T                                  # [chunk, bank_size]
        top_sim, top_idx = similarity.topk(k, dim=1)                 # k nearest bank images
        neighbor_labels = bank_labels[top_idx]                       # [chunk, k]
        vote_weight = torch.exp(top_sim / temperature)

        class_scores = torch.zeros(len(chunk), num_classes)
        class_scores.scatter_add_(1, neighbor_labels, vote_weight)   # sum weights per class

        true_labels = query_labels[start:start + chunk_size]
        top5_classes = class_scores.topk(min(5, num_classes), dim=1).indices
        correct_top1 += (top5_classes[:, 0] == true_labels).sum().item()
        correct_top5 += (top5_classes == true_labels[:, None]).any(dim=1).sum().item()

    return {"top1": correct_top1 / len(queries), "top5": correct_top5 / len(queries)}
