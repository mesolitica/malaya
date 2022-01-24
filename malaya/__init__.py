# Malaya Natural Language Toolkit
#
# Copyright (C) 2019 Malaya Project
# Licensed under the MIT License
# Author: huseinzol05 <husein.zol05@gmail.com>
# URL: <https://malaya.readthedocs.io/>
# For license information, see https://github.com/huseinzol05/Malaya/blob/master/LICENSE

from malaya_boilerplate.utils import get_home

version = '4.7'
bump_version = '4.7.1'
__version__ = bump_version

package = 'malaya'
url = 'https://f000.backblazeb2.com/file/malaya-model/'
__home__, _ = get_home(package=package, package_version=version)


