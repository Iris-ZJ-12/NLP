# encoding: utf-8
'''
@author: zyl
@file: xml_extractor.py
@time: 2021/9/16 14:30
@desc: parse xml, see: https://lxml.de/parsing.html
'''

import gzip
import os
from io import BytesIO

from lxml import etree


class XMLExtractor:
    def __init__(self):
        pass

    @staticmethod
    def read_xml(file_or_text, events=('end',), encoding='UTF-8', tag=None):
        if os.path.exists(file_or_text):
            if file_or_text.endswith('.xml.gz'):
                data = etree.iterparse(gzip.GzipFile(file_or_text), events=events, encoding=encoding, tag=tag)
            else:  # .xml
                data = etree.iterparse(file_or_text, events=events, encoding=encoding, tag=tag)
        else:
            if isinstance(file_or_text, str):
                data = etree.iterparse(BytesIO(bytes(file_or_text, encoding="utf8")), events=events, encoding=encoding,
                                       tag=tag)
            else:
                data = etree.iterparse(BytesIO(file_or_text), events=events, encoding=encoding, tag=tag)
        return data  # iterator

    @staticmethod
    def parse_big_xml_gz(file, process_element_func, tag='{http://uniprot.org/uniprot}entry',
                         events=('end',), encoding='UTF-8', *args, **kwargs):
        """
        Deal with a big .xml.gz file (>100G) which can't be loaded into the memory.
        Args:
            file: a big .xml.gz file, str
            process_element_func: the function for processing the element in each dt,the first parameter is the element in xml
            tag: xml tag,usually： namespace+element_node
            *args: the process_element_func's parameters
            **kwargs:the process_element_func's parameters
        Returns:
        """
        data = XMLExtractor.read_xml(file, events, encoding, tag)

        for event, elem in data:
            process_element_func(elem, *args, **kwargs)

            # del nodes ---must have
            elem.clear()
            for ancestor in elem.xpath('ancestor-or-self::*'):
                while ancestor.getprevious() is not None:
                    del ancestor.getparent()[0]
        del data


if __name__ == '__main__':
    xml_text = """
    <bookstore>
        <book category="CHILDREN">
            <title>Harry Potter</title> 
            <author>J K. Rowling</author> 
            <year>2005</year> 
            <price>29.99</price> 
        </book>
        <book category="WEB">
            <title>Learning XML</title> 
            <author>Erik T. Ray</author> 
            <year>2003</year> 
            <price>39.95</price> 
        </book>
    </bookstore>
"""

    def extract_title(elem):
        title = elem.xpath('./*[local-name()="title"]/text()')
        print(f'title：{title}')


    XMLExtractor.parse_big_xml_gz(xml_text, process_element_func=extract_title,
                                  tag='book')
