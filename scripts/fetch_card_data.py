#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Author: David J. Elkind
# Creation date: 2021-07-18 (year-month-day)

"""
"""
import argparse
import pathlib
import json
from mtgsdk import Card


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outfile",
        type=pathlib.Path,
        required=True,
        help="where to write the data",
    )
    parser.add_argument(
        "-t",
        "--test_run",
        action="store_true",
        help="set this flag to only query a small number of results",
    )
    return parser.parse_args()


exclude = ["foreign_names"]


def card2dict(single_card):
    assert isinstance(single_card, Card)
    tmp = {k: v for k, v in single_card.__dict__.items() if k not in exclude}
    return tmp


def resp_unique(mtg_resp):
    cards = [card2dict(c) for c in mtg_resp]
    out = dict()
    for card in cards:
        if card["name"] not in out:
            out.update({card["name"]: card})
        # elif card["release_date"] > out["card"]["release_date"]:
        #     out.update({card["name"]: card})
    return out


if __name__ == "__main__":
    args = parse_args()
    assert not args.outfile.is_file()
    if args.test_run:
        resp_legendary = (
            Card.where(supertypes="legendary").where(page=1).where(pageSize=10).all()
        )
        resp_planeswalker = (
            Card.where(types="planeswalker").where(page=1).where(pageSize=10).all()
        )
        resp_creatures = (
            Card.where(types="creature").where(page=1).where(pageSize=10).all()
        )
    else:
        resp_legendary = Card.where(supertypes="legendary").all()
        resp_planeswalker = Card.where(types="planeswalker").all()
        resp_creatures = Card.where(types="creature").all()
    unique_cards = resp_unique(resp_legendary)
    if len(unique_cards) < 100:
        print(unique_cards)
        print(unique_cards.keys())
    unique_cards.update(resp_unique(resp_planeswalker))
    unique_cards.update(resp_unique(resp_creatures))
    print(f"There are {len(unique_cards)} cards in the query results.")
    with open(args.outfile, "w") as f:
        json.dump(unique_cards, f)
