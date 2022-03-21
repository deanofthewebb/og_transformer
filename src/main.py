from enum import Enum
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
import torch.nn as nn
from layers.transformers import Transformer
from torchtext.datasets import Multi30k, IWSLT2017
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchmetrics import BLEUScore
from torch.utils.data import DataLoader
import math as m
import statistics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import wandb

BATCH_SIZE = 128
MAX_SEQ_LEN = 40
MIN_EPOCHS = 15
PROJECT_NAME = "og_transformer_IWSLT2017_colab"
LR = 0.000035
WANDB_LOGGER = WandbLogger(project=PROJECT_NAME)



class SpecialTokens(Enum):
    SRC = "src"
    TRG = "trg"
    PAD = "<pad>"
    ENGLISH = "en"
    GERMAN = "de"
    UNKOWN = "<unk>"
    START_OF_SENTENCE = "<sos>"
    END_OF_SENTENCE = "<eos>"
    BASIC_ENGLISH = "basic_english"

    def __str__(self):
        return str(self.value)


class TransformerTrainer(pl.LightningModule):
    def __init__(self, src_vocab: Vocab, trg_vocab: Vocab, warmup_steps=15000, d_model=512, d_ff=2048, 
                 num_layers=6, num_heads=8, device="cuda", dropout=0.1, lr=LR, max_epochs=500,
                 min_epochs=MIN_EPOCHS, warmup_epochs=100, batch_size=BATCH_SIZE):
        super().__init__()

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device_ = device
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_epochs = max_epochs
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.metric = BLEUScore()

        self.model = Transformer(
            src_vocab_len=len(self.src_vocab),
            trg_vocab_len=len(self.trg_vocab),
            d_model=self.d_model,
            d_ff=d_ff,
            num_layers=num_layers,
            num_heads=num_heads,
            src_pad_idx=self.src_vocab.__getitem__(str(SpecialTokens.PAD)),
            trg_pad_idx=self.trg_vocab.__getitem__(str(SpecialTokens.PAD)),
            dropout=dropout,
            device=self.device_,
            efficient_mha=True
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_vocab.__getitem__(str(SpecialTokens.PAD)))
        self.scheduler = None
        print(self.model)
        WANDB_LOGGER.watch(self.model)
    
    def training_step(self, batch, batch_idx):
        src = batch[0].to(self.device_)
        trg = batch[1].to(self.device_)

        trg_input = trg[:, :-1]
        ys = trg[:, 1:].reshape(-1)

        logits = self.model(src, trg_input)

        loss = self.criterion(logits.reshape(-1, len(self.trg_vocab)), ys)
        
        self.log("training loss", loss)

        self.change_lr_in_optimizer()

        if batch_idx == 0:
            for idx in range(0, len(src), 200):
                print("(train)  SRC:\t", self.clean_and_print_tokens(src[idx], str(SpecialTokens.SRC)))
                print("(train)  TRG:\t", self.clean_and_print_tokens(trg[idx], str(SpecialTokens.TRG)))
                print("(train) PRED:\t", self.clean_and_print_tokens(torch.argmax(logits[idx], dim=-1), str(SpecialTokens.TRG)))
                print("")

        return loss

    def validation_step(self, batch, batch_idx):
        src = batch[0].to(self.device_)
        trg = batch[1].to(self.device_)
        trg_input = trg[:, :-1]

        logits = self.model(src, trg_input)
        # logits.shape = [B x TRG_seq_len x TRG_vocab_size]

        ys = trg[:, 1:].reshape(-1)
        val_loss = self.criterion(logits.reshape(-1, len(self.trg_vocab)), ys)

        self.log("validation loss", val_loss)

        bleu_scores = []
        for idx in range(len(src)):
            trg = self.clean_and_print_tokens(trg[idx], str(SpecialTokens.TRG))
            pred = self.clean_and_print_tokens(torch.argmax(logits[idx], dim=-1), str(SpecialTokens.TRG))
            bleu_scores.append(self.metric(trg.split(), pred.split()).item())
            if idx % 250 == 0:
                print(" SRC:\t", self.clean_and_print_tokens(src[idx], str(SpecialTokens.SRC)))
                print(" TRG:\t", trg)
                print("PRED:\t", pred)
                print("-------")
        
        mean_bleu = statistics.mean(bleu_scores)
            

        print("Val Loss:", val_loss)
        print("BLEU Score:", mean_bleu)
        # self.scheduler.step()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # self.scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=MIN_EPOCHS, max_epochs=self.max_epochs)
                        
        return [optimizer]

    # Google Transformers lr schedluer
    def change_lr_in_optimizer(self):
        min_arg1 = m.sqrt(1/(self.global_step + 1))
        min_arg2 = self.global_step * (self.warmup_steps**-1.5)
        lr = m.sqrt(1/self.d_model) * min(min_arg1, min_arg2)
        self.trainer.lightning_optimizers[0].param_groups[0]['lr'] = lr
        self.log("learning rate", lr)


    def clean_and_print_tokens(self, tokens, src_or_trg):
        if src_or_trg == str(SpecialTokens.SRC):
            vocab = self.src_vocab
        elif src_or_trg == str(SpecialTokens.TRG):
            vocab = self.trg_vocab

        return " ".join(vocab.lookup_tokens(tokens.tolist()))


if __name__ == "__main__":
    device = ("cuda:0" if torch.cuda.is_available else "cpu")
    torch.cuda.empty_cache()

    train_iter, val_iter, test_iter = IWSLT2017(language_pair=(str(SpecialTokens.GERMAN), str(SpecialTokens.ENGLISH)))
    src_tokenizer = get_tokenizer(str(SpecialTokens.BASIC_ENGLISH))
    trg_tokenizer = get_tokenizer(str(SpecialTokens.BASIC_ENGLISH))

    def yield_tokens(data_iter, src_or_trg):
        for batch in data_iter:
            if src_or_trg == str(SpecialTokens.SRC):
                yield src_tokenizer(batch[0])
            elif src_or_trg == str(SpecialTokens.TRG):
                yield trg_tokenizer(batch[1])

    src_vocab = build_vocab_from_iterator(yield_tokens(train_iter, str(SpecialTokens.SRC)), specials=[str(SpecialTokens.UNKOWN), str(SpecialTokens.PAD), str(SpecialTokens.START_OF_SENTENCE), str(SpecialTokens.END_OF_SENTENCE)])
    src_vocab.set_default_index(src_vocab[str(SpecialTokens.UNKOWN)])

    train_iter, val_iter, test_iter = IWSLT2017(language_pair=(str(SpecialTokens.GERMAN), str(SpecialTokens.ENGLISH)))

    trg_vocab = build_vocab_from_iterator(yield_tokens(train_iter, str(SpecialTokens.TRG)), specials=[str(SpecialTokens.UNKOWN), str(SpecialTokens.PAD), str(SpecialTokens.START_OF_SENTENCE), str(SpecialTokens.END_OF_SENTENCE)])
    trg_vocab.set_default_index(trg_vocab[str(SpecialTokens.UNKOWN)])

    train_iter, val_iter, test_iter = IWSLT2017(language_pair=(str(SpecialTokens.GERMAN), str(SpecialTokens.ENGLISH)))

    

    def pad_to_max(tokens):
        return tokens[:MAX_SEQ_LEN] + [str(SpecialTokens.PAD)] * max(0, MAX_SEQ_LEN - len(tokens))

    def collate_fn(batch):
        # batch = [(<src1>, <trg1>), (<src2>, <trg2>), ...]
        sources = []
        targets = []
        for pair in batch:
            src = pair[0]
            trg = pair[1]

            tokenized_src = src_vocab(pad_to_max(src_tokenizer(f"{SpecialTokens.START_OF_SENTENCE} " + src + f" {SpecialTokens.END_OF_SENTENCE}")))
            tokenized_trg = trg_vocab(pad_to_max(trg_tokenizer(f"{SpecialTokens.START_OF_SENTENCE} " + trg + f" {SpecialTokens.END_OF_SENTENCE}")))

            sources.append(tokenized_src)
            targets.append(tokenized_trg)

        sources = torch.tensor(sources, dtype=torch.long)
        targets = torch.tensor(targets, dtype=torch.long)
        return sources, targets

    dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn)
    val_dataloader = DataLoader(list(val_iter), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn)
    test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=collate_fn)
    
    
    transformer = TransformerTrainer(src_vocab, trg_vocab, device=device)

    tb_logger = TensorBoardLogger("tb_logs", name=PROJECT_NAME)
    
    trainer = pl.Trainer(gpus=1, min_epochs=MIN_EPOCHS, logger=[WANDB_LOGGER, tb_logger], 
                        callbacks=[EarlyStopping(monitor="validation loss", patience=15, mode="min")])
    trainer.fit(transformer, dataloader, val_dataloader)