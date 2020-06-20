UnicodeDecodeError: ‘charmap’ codec can’t decode byte
-----------------------------------------------------

To solve this,

Windows Settings > Administrative language settings > Change system
locale.

Checked Beta: Use Unicode UTF-8 for worldwide language support.

Restarted, everything works well.

Full dicussion check `issue
25 <https://github.com/huseinzol05/Malaya/issues/25>`__.

Unable to use any T5 models
---------------------------

T5 depends on tensorflow-text, currently there is no official
tensorflow-text binary released for Windows. So no T5 model for Windows
users.

Lack development on Windows
---------------------------

Malaya been tested on modern UNIX / LINUX systems that supported
tensorflow binary. Windows support not our development path right now
(we don’t have any Windows machine here).

