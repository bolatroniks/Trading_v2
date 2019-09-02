# -*- coding: utf-8 -*-

import datapackage
import pandas as pd

data_url = 'https://datahub.io/core/s-and-p-500/datapackage.json'

# to load Data Package into storage
package = datapackage.Package(data_url)

# to load only tabular data
resources = package.resources
for resource in resources:
    if resource.tabular:
        data = pd.read_csv(resource.descriptor['path'])
        print (data)