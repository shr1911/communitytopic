# Community Topic
## Introduction
- **What is Community Topic**?

In this repository we present our novel method called Community Topic for Topic Modelleling as a Pypi library. Our algorithm, Community Topic, is based on mining communities of terms from term-occurrence networks extracted from the documents. In addition to providing interpretable collections of terms as topics, the network representation provides a natural topic structure. The topics form a network, so topic similarity is inferred from the weights of the edges between them. Super-topics can be found by iteratively applying community detection on the topic network, grouping similar topics together. Sub-topics can be found by iteratively applying community detection on a single topic community. This can be done dynamically, with the user or conversation agent moving up and down the topic hierarchy as desired.

- **What problem does it solve? & Who is it for?**

Unfortunately, the most popular topic models in use today do not provide a suitable topic structure for these purposes and the state-of-the-art models based on neural networks suffer from many of the same drawbacks while requiring specialized hardware and many hours to train. This makes Community Topic an ideal topic modelling algorithm for both applied research and practical applications like conversational agents.

## Installation

- System requirement

      Python >= 3.6
      commodity hardware
      setuptools~=67.6.0
      spacy~=3.5.0
      numpy~=1.21.5
      gensim~=4.2.0
      networkx~=2.8.4
      igraph~=0.10.4


- Installation Option

The easy way to install CommunityTopic is:

      pip install communitytopic

## Usage
Here are some examples of how to use Community Topic:

- What are the main use cases for Community Topic?
- How can the library be used?
- What are the key features of the library?
- Are there any best practices for using the library?

## Vizualization

- Flat topics
- Hierarchical topics

## Contributing
- How can users get involved?
- What are the guidelines for contributing?
- Are there any coding standards or guidelines that contributors should follow?
- What are the best ways to submit issues and feature requests?

## License
- What license is the library released under?
- What are the terms of the license?
- Are there any restrictions on how the library can be used or distributed?

## Credits
- Who created the library?
- Who are the contributors to the project?
- What sources or tools were used in the creation of the library?

## Contact
- Who should users contact for support or questions?
- What is the best way to reach the project maintainer or team?
- Are there any mailing lists, chat channels, or other communication channels that users can use to connect with the community?

## Conclusion
- What are the main benefits of using Community Topic?
- Why should users choose this library over other alternatives?
- What is the future of the library, and how can users get involved?
