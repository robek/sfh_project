from django.conf.urls.defaults import patterns, include, url

# Uncomment the next two lines to enable the admin:
# from django.contrib import admin
# admin.autodiscover()

urlpatterns = patterns('sfh.views',
    url(r'^$', 'index', name='index'),
    url(r'^data$', 'data', name='data'),
    url(r'^train$', 'train', name='train'),
    url(r'^tide$', 'tide', name='tide'),
    url(r'^mix$', 'mix', name='mix'),
    url(r'^dummy$', 'dummy', name='dummy'),
    url(r'^cross$', 'cross', name='cross'),
    url(r'^show$', 'show', name='show'),
    url(r'^delTide$', 'delTide', name='delTide'),
    url(r'^graph$', 'graph', name='graph'),
    url(r'^pearson$', 'pearson', name='pearson'),
    
    # Examples:
    # url(r'^$', 'sfh_project.views.home', name='home'),
    # url(r'^sfh_project/', include('sfh_project.foo.urls')),

    # Uncomment the admin/doc line below to enable admin documentation:
    # url(r'^admin/doc/', include('django.contrib.admindocs.urls')),

    # Uncomment the next line to enable the admin:
    # url(r'^admin/', include(admin.site.urls)),
)
