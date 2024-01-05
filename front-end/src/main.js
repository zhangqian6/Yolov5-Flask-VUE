// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import VueRouter from 'vue-router'
import axios from 'axios'
import Element from 'element-ui'
import echarts from "echarts";
import Content from "./components/Content.vue";
import Header from  "./components/Header.vue";
import Footer from "./components/Footer.vue"

Vue.prototype.$echarts = echarts;     //定义全局变量echarts
import '../node_modules/element-ui/lib/theme-chalk/index.css'
import '../src/assets/style.css'
import './theme/index.css'

Vue.use(Element)  //使'element-ui'中的组件能被vue使用
Vue.config.productionTip = false  //true它会显示你生产模式的消息。所以，在开发环境下，保持默认设置false即可
//Router进行组件布局
Vue.use(VueRouter)
Vue.prototype.$http = axios//HTTP请求获取数据。这是一种通用的方式，允许你在Vue组件中方便地进行异步操作。
// 创建路由实例
const router = new VueRouter({
    routes: [{
        //路由规则
        path: "/App",
        component: App,  //, meta: {title: "眼疾辅助诊断系统"}
        children:[
            {
                path:'header',
                component: Header,
            },
            {
                path:'footer',
                component: Footer,
            },
            {
                path:'content',
                component: Content,
            },
        ]
    }
    ],
    mode: "history"
})

// 全局注册组件
Vue.component("App", App);

/* eslint-disable no-new */
new Vue({
    el: '#app',
    data:{
        footerContent:'这是footer的内容'
    },
    router,
    render: h => h(App)
})







