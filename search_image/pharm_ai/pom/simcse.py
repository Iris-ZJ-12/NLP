from typing import List
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import losses
from torch.utils.data import DataLoader


def unsup_simcse(model: SentenceTransformer, train_sentences: List[str]) -> None:
    batch_size = 64
    lr = 3e-5
    warmup_steps = int(len(train_sentences) / batch_size * 0.1)
    
    train_data = [InputExample(texts=[s, s]) for s in train_sentences]
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=1,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
        optimizer_params={'lr': lr}
    )
