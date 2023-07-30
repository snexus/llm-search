## 30-07-2023

### Changelog:

* Code cleaning and refactoring

* Improvements to the markdown parser:
    - Added options to clean markdown before processing, which includes removing image links and extra new lines.
    - Implemented the ability to extract custom metadata and attach it to every output text chunk.

* Enhancements to document management:
    - Now supports including multiple document paths (refer to the new format of config.yaml for details).
    - Added the ability to perform multiple search/replace substitutions for the output paths.

* Experimental web interface (Streamlit):
