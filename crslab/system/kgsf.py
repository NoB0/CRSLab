# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/24, 2021/1/3
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
from typing import Any, List, Tuple

import torch
from loguru import logger

from crslab.data import dataset_language_map, get_dataloader
from crslab.evaluator.metrics.base import AverageMetric
from crslab.evaluator.metrics.gen import PPLMetric
from crslab.system.base import BaseSystem
from crslab.system.utils.functions import ind2txt


class KGSFSystem(BaseSystem):
    """This is the system for KGSF model"""

    def __init__(
        self,
        opt,
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        vocab,
        side_data,
        restore_system=False,
        interact=False,
        debug=False,
        tensorboard=False,
    ):
        """

        Args:
            opt (dict): Indicating the hyper parameters.
            train_dataloader (BaseDataLoader): Indicating the train dataloader of corresponding dataset.
            valid_dataloader (BaseDataLoader): Indicating the valid dataloader of corresponding dataset.
            test_dataloader (BaseDataLoader): Indicating the test dataloader of corresponding dataset.
            vocab (dict): Indicating the vocabulary.
            side_data (dict): Indicating the side data.
            restore_system (bool, optional): Indicating if we store system after training. Defaults to False.
            interact (bool, optional): Indicating if we interact with system. Defaults to False.
            debug (bool, optional): Indicating if we train in debug mode. Defaults to False.
            tensorboard (bool, optional) Indicating if we monitor the training performance in tensorboard. Defaults to False.

        """
        super(KGSFSystem, self).__init__(
            opt,
            train_dataloader,
            valid_dataloader,
            test_dataloader,
            vocab,
            side_data,
            restore_system,
            interact,
            debug,
            tensorboard,
        )

        self.ind2tok = vocab["ind2tok"]
        self.end_token_idx = vocab["end"]
        self.item_ids = side_data["item_entity_ids"]

        self.pretrain_optim_opt = self.opt["pretrain"]
        self.rec_optim_opt = self.opt["rec"]
        self.conv_optim_opt = self.opt["conv"]
        self.pretrain_epoch = self.pretrain_optim_opt["epoch"]
        self.rec_epoch = self.rec_optim_opt["epoch"]
        self.conv_epoch = self.conv_optim_opt["epoch"]
        self.pretrain_batch_size = self.pretrain_optim_opt["batch_size"]
        self.rec_batch_size = self.rec_optim_opt["batch_size"]
        self.conv_batch_size = self.conv_optim_opt["batch_size"]

        self.language = dataset_language_map[self.opt["dataset"]]

    def rec_evaluate(self, rec_predict, item_label):
        rec_predict = rec_predict.cpu()
        rec_predict = rec_predict[:, self.item_ids]
        _, rec_ranks = torch.topk(rec_predict, 50, dim=-1)
        rec_ranks = rec_ranks.tolist()
        item_label = item_label.tolist()
        for rec_rank, item in zip(rec_ranks, item_label):
            item = self.item_ids.index(item)
            self.evaluator.rec_evaluate(rec_rank, item)

    def conv_evaluate(self, prediction, response):
        prediction = prediction.tolist()
        response = response.tolist()
        for p, r in zip(prediction, response):
            p_str = ind2txt(p, self.ind2tok, self.end_token_idx)
            r_str = ind2txt(r, self.ind2tok, self.end_token_idx)
            self.evaluator.gen_evaluate(p_str, [r_str])

    def step(self, batch, stage, mode):
        batch = [ele.to(self.device) for ele in batch]
        if stage == "pretrain":
            info_loss = self.model.forward(batch, stage, mode)
            if info_loss is not None:
                self.backward(info_loss.sum())
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "rec":
            rec_loss, info_loss, rec_predict = self.model.forward(batch, stage, mode)
            if info_loss:
                loss = rec_loss + 0.025 * info_loss
            else:
                loss = rec_loss
            if mode == "train":
                self.backward(loss.sum())
            else:
                self.rec_evaluate(rec_predict, batch[-1])
            rec_loss = rec_loss.sum().item()
            self.evaluator.optim_metrics.add("rec_loss", AverageMetric(rec_loss))
            if info_loss:
                info_loss = info_loss.sum().item()
                self.evaluator.optim_metrics.add("info_loss", AverageMetric(info_loss))
        elif stage == "conv":
            if mode != "test":
                gen_loss, pred = self.model.forward(batch, stage, mode)
                if mode == "train":
                    self.backward(gen_loss.sum())
                else:
                    self.conv_evaluate(pred, batch[-1])
                gen_loss = gen_loss.sum().item()
                self.evaluator.optim_metrics.add("gen_loss", AverageMetric(gen_loss))
                self.evaluator.gen_metrics.add("ppl", PPLMetric(gen_loss))
            else:
                pred = self.model.forward(batch, stage, mode)
                self.conv_evaluate(pred, batch[-1])
        else:
            raise

    def pretrain(self):
        self.init_optim(self.pretrain_optim_opt, self.model.parameters())

        for epoch in range(self.pretrain_epoch):
            self.evaluator.reset_metrics()
            logger.info(f"[Pretrain epoch {str(epoch)}]")
            for batch in self.train_dataloader.get_pretrain_data(
                self.pretrain_batch_size, shuffle=False
            ):
                self.step(batch, stage="pretrain", mode="train")
            self.evaluator.report()

    def train_recommender(self):
        self.init_optim(self.rec_optim_opt, self.model.parameters())

        for epoch in range(self.rec_epoch):
            self.evaluator.reset_metrics()
            logger.info(f"[Recommendation epoch {str(epoch)}]")
            logger.info("[Train]")
            for batch in self.train_dataloader.get_rec_data(
                self.rec_batch_size, shuffle=False
            ):
                self.step(batch, stage="rec", mode="train")
            self.evaluator.report(epoch=epoch, mode="train")
            # val
            logger.info("[Valid]")
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_rec_data(
                    self.rec_batch_size, shuffle=False
                ):
                    self.step(batch, stage="rec", mode="val")
                self.evaluator.report(epoch=epoch, mode="val")
                # early stop
                metric = (
                    self.evaluator.rec_metrics["hit@1"]
                    + self.evaluator.rec_metrics["hit@50"]
                )
                if self.early_stop(metric):
                    break
        # test
        logger.info("[Test]")
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_rec_data(
                self.rec_batch_size, shuffle=False
            ):
                self.step(batch, stage="rec", mode="test")
            self.evaluator.report(mode="test")

    def train_conversation(self):
        if os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
            self.model.freeze_parameters()
        else:
            self.model.module.freeze_parameters()
        self.init_optim(self.conv_optim_opt, self.model.parameters())

        for epoch in range(self.conv_epoch):
            self.evaluator.reset_metrics()
            logger.info(f"[Conversation epoch {str(epoch)}]")
            logger.info("[Train]")
            for batch in self.train_dataloader.get_conv_data(
                batch_size=self.conv_batch_size, shuffle=False
            ):
                self.step(batch, stage="conv", mode="train")
            self.evaluator.report(epoch=epoch, mode="train")
            # val
            logger.info("[Valid]")
            with torch.no_grad():
                self.evaluator.reset_metrics()
                for batch in self.valid_dataloader.get_conv_data(
                    batch_size=self.conv_batch_size, shuffle=False
                ):
                    self.step(batch, stage="conv", mode="val")
                self.evaluator.report(epoch=epoch, mode="val")
        # test
        logger.info("[Test]")
        with torch.no_grad():
            self.evaluator.reset_metrics()
            for batch in self.test_dataloader.get_conv_data(
                batch_size=self.conv_batch_size, shuffle=False
            ):
                self.step(batch, stage="conv", mode="test")
            self.evaluator.report(mode="test")

    def fit(self):
        self.pretrain()
        self.train_recommender()
        self.train_conversation()

    def interact(self):
        """Interact with the system."""
        self.init_interact()
        input_text = self.get_input(self.language)
        while not self.finished:
            data_input = self.process_input(input_text)
            # Recommendation
            recommender_input = data_input[0]
            rec_predict = self.model.forward(
                recommender_input, stage="rec", mode="test"
            )
            rec_predict = rec_predict.cpu()[0]
            rec_predict = rec_predict[self.item_ids]
            _, rec_ranks = torch.topk(rec_predict, 10, dim=-1)
            item_ids = [self.item_ids[idx] for idx in rec_ranks.tolist()]
            first_item_id = item_ids[:1]
            self.update_context("rec", entity_ids=first_item_id, item_ids=first_item_id)
            recommended_entities = [
                self.vocab["id2entity"].get(item_id, "UNK") for item_id in item_ids
            ]
            print(f"[Recommend]:\n{';'.join(recommended_entities)}")

            # Conversation
            conversation_input = data_input[1]
            preds = self.model.forward(conversation_input, stage="conv", mode="test")
            preds = preds.tolist()[0]
            response = ind2txt(preds, self.ind2tok, self.end_token_idx)
            token_ids, entity_ids, word_ids, movie_ids = self.convert_to_ids(response)
            self.update_context("conv", token_ids, entity_ids, movie_ids, word_ids)
            print(f"[Response]:\n{response}")

            # TODO: Merge the two stages to include recommendation in generated
            # response

            input_text = self.get_input(self.language)

    def process_input(
        self,
        input_text: str,
    ) -> List[Any]:
        """Process the input utterance.

        Args:
            input_text: Incoming utterance.

        Returns:
            List with input data for recommendation and conversation.
        """
        token_ids, entity_ids, word_ids, movie_ids = self.convert_to_ids(input_text)
        data_input = []
        for stage in ["rec", "conv"]:
            self.update_context(stage, token_ids, entity_ids, movie_ids, word_ids)

            data = {
                "context_tokens": self.context[stage]["context_tokens"],
                "context_entities": self.context[stage]["context_entities"],
                "context_words": self.context[stage]["context_words"],
            }
            dataloader = get_dataloader(self.opt, data, self.vocab)
            if stage == "rec":
                data = dataloader.rec_interact(data)
            elif stage == "conv":
                data = dataloader.conv_interact(data)

            data = self._prepare_tensor_data(data)
            data_input.append(data)
        return data_input

    def _prepare_tensor_data(self, data) -> List[Any]:
        """Sends tensors to the device.

        Args:
            data: Input data.

        Returns:
            Input data with tensors on the device.
        """
        data = [
            ele.to(self.device) if isinstance(ele, torch.Tensor) else ele
            for ele in data
        ]
        return data

    def convert_to_ids(self, utterance: str) -> Tuple[List[int]]:
        """Converts utterance to token ids, entities ids, and word ids.

        Args:
            utterance: Incoming utterance.

        Returns:
            Tuple with token ids, entities ids, word ids, and movie ids.
        """
        tokens = self.tokenize(utterance, "nltk")
        entities = self.link(tokens, self.side_data["entity_kg"]["entity"])
        words = self.link(tokens, self.side_data["word_kg"]["entity"])

        token_ids = [
            self.vocab["tok2ind"].get(token, self.vocab["unk"]) for token in tokens
        ]
        entity_ids = [
            self.vocab["entity2id"][entity]
            for entity in entities
            if entity in self.vocab["entity2id"]
        ]
        movie_ids = [
            entity_id for entity_id in entity_ids if entity_id in self.item_ids
        ]
        word_ids = [
            self.vocab["word2id"][word]
            for word in words
            if word in self.vocab["word2id"]
        ]
        return token_ids, entity_ids, word_ids, movie_ids
