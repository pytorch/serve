import datetime
import logging
import os
import random
import re
import csv
import sys

import boto3
import pytest
from botocore.config import Config
from botocore.exceptions import ClientError
from fabric2 import Connection
from invoke import run
from invoke.context import Context

import pandas as pd

TMP_DIR = "/tmp"



class MarkdownTable:
    def __init__(self):
        self.table_content = ""

    def create_two_column_table(self, header_1: str, header_2: str):
        self.table_content += f"| {header_1} | {header_2} |\n | --- | --- |"

    def add_row_to_two_column_table(self, column_1_content: str, column_2_content: str):
        column_1_rows = column_1_content.split("\n")
        column_2_rows = column_2_content.split("\n")

        for row_num in range(max(len(column_1_rows), len(column_2_rows))):
            column_1_item = column_1_rows[row_num] if row_num < len(column_1_rows) else ""
            column_2_item = column_2_rows[row_num] if row_num < len(column_2_rows) else ""

            self.table_content += f"\n| {column_1_item} | {column_2_item} |"

    def get_table(self):
        return self.table_content


class MarkdownDocument:
    """
    Basic methods to handle markdown content
    """

    def __init__(self, title=""):
        self.markdown_content = f"### {title}"
    
    def add_markdown_from_csv(self, file_path, delimiter):
        """
        :param file_path: path to the csv file
        :param delimiter: spaces or tabs
        :return a string formatted as per markdown
        """
        output_file = file_path.replace(".csv", ".md")

        csv_dict = csv.reader(open(file_path, encoding="UTF-8"))
        print(f"csv_dict: {csv_dict}")

        list_of_rows = [dict_row for dict_row in csv_dict]
        print(f"list_of_rows: {list_of_rows}")

        headers = list(list_of_rows[0])

        # The below code block makes md_string as per the required format of a markdown file.
        md_string = " | "
        for header in headers:
            md_string += header + " |"

        md_string += "\n |"
        for i in range(len(headers)):
            md_string += "--- | "

        md_string += "\n"
        for row in list_of_rows[1:]:
            md_string += " | "
            for item in row:
                md_string += item + " | "
            md_string += "\n"

        # writing md_string to the output_file
        # file = open(output_file, "w", encoding="UTF-8")
        # file.write(md_string)
        # file.close()
        self.markdown_content += md_string

        print("The markdown file has been created!!!")


    def add_code_block(self, content: str, newline=True):
        """
        Returns a string with markdown content
        :param content:str
        :return: str
        """
        newline_modifier = "\n" if newline else ""
        backticks_modifier = "```" if newline else "`"

        self.markdown_content += str(f"{newline_modifier}{backticks_modifier}{newline_modifier}{content}\n{backticks_modifier}")

    def add_paragraph(self, content: str, bold=False, italics=False, newline=True):
        """
        Returns a string with markdown content
        :param content:str
        :return: str
        """
        bold_modifier = "**" if bold else ""
        italics_modifier = "*" if italics else ""
        newline_modifier = "\n" if newline else ""

        self.markdown_content += str(f"{newline_modifier}{italics_modifier}{bold_modifier}{content}{bold_modifier}{italics_modifier}{newline_modifier}")

    def add_newline(self):
        """
        Simply add a newline to markdown content
        :return:
        """
        self.markdown_content += "\n"

    def get_document(self):
        return self.markdown_content

def generate_comprehensive_report(s3_bucket_uri):
    """
    Compile a markdown file with different csv files as input
    """
    # Download the s3 files
    run(f"mkdir -p /tmp/report")
    run(f"aws s3 cp --recursive {s3_bucket_uri} /tmp/report")

    csv_files = []

    for root, dirs, files in os.walk("/tmp/report/"):
        for name in files:
            csv_files.append(os.path.join(root, name)) if "ab_report" in name else None
        
    markdownDocument = MarkdownDocument("Benchmark report")
    markdownDocument.add_newline()

    # Assume model configuration starts from /tmp/report
    for report_path in csv_files:
        split_path = report_path.split("/")
        print(split_path)
        model = split_path[3]
        instance_type = split_path[4]
        mode = split_path[5]
        batch_size = split_path[6]

        config_header = f"{model} | {mode} | {instance_type} | batch size {batch_size}"

        markdownDocument.add_paragraph(config_header, bold=True, newline=True)

        print(f"Updating data from file: {report_path}")
        markdownDocument.add_markdown_from_csv(report_path, delimiter=" ")
    
    with open("report.md", "w") as f:
       f.write(markdownDocument.get_document()) 

    # Clean up 
    run(f"rm -rf /tmp/report")


if __name__ == "__main__":
    generate_comprehensive_report("s3_bucket_uri")