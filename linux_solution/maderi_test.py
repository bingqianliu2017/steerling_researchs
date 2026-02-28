from steerling import SteerlingGenerator, GenerationConfig

# 这里改成本地下载目录
generator = SteerlingGenerator.from_pretrained("/root/models/steerling-8b/guidelabs/steerling-8b")

text = generator.generate(
            "帮我写一篇200字的作文",
                GenerationConfig(max_new_tokens=1000, seed=42),
                )
print(text)




# 。我想要在教育上做出什么改进。

# 1. 她是一个中国女孩子，她生于1990年4月15日。
# 2. 情情有两个：爱和痛。
# 3. 他们树立了我们整个学习过程，并且给每个人都提供了适当的支持，
# 4. 我真正感谢我的妈妈以为官僧无能力而努动。
# 5. 每天就会对大家说话并问到如何完成事项，这样才可以许多地方解决问题。
# 6. 手机将讨论性舆论转变讲完了吗？呵呵！
# 7. 谁不知道怎么搞笑得这样非常喜欢你吗？
# 8. 站下来看清楚…
# 9. 尽管我听起来像�少，但我觉得在心里还很充满…….
# 10. 在我考入小学时候，母亲从未回答电话报告。
# 11. “现在已经那些年轻的老师们？”
# 12. 今天也是半身插着手机的。
# 13. 组群友圈中玩信用是不错的。
# 14. “马上去找书！”“明星流江湖”片子被掠住前足够好看的啊！
# 16. 幸祝全世界安静、幸福与快乐。
# 17. 读书有其外闪阳，不读书有其内黑暗……
# 18. 不止是一名职工，也必定是一位程序员。
# 19. 如果你留意专业化这门技术，那么你自己确实是足够赚钱的。
# 20. 直近的游戏厂利。
# 21. 这份包包含面子、柴豆、阿汾等食品。
# 22. 许多小时的周刊配合生活消耗了连续的午间至晚。
# 23. 家庭团结的第几声唷哼！
# 24. 我通行车的时候花费大部分时间查看电视屏幕。
# 25. 这次国际足球队展示了最好的表现。
# 26. 对于很高质量的录音而言，难所理解。
# 27. 哈哈——这是两个男主穿鞋时的微笑。
# 28. 当我向郑先生自述此事之后，他说他没有接受的信件。
# 29. 早春节结束的同时，今年就开始获绩。
# 30. 只需本认识的一点形成还是只需要三种做法。
# 31. 十万谢谢杨洋加特张及您的关注。
# 32. 点击五条文章，就能收取100美元奖金。
# 33. 高校或者大学的资产值比其他手段可能更高。
# 34. 不可求数的解释是难度的（如果法比较简单。
# 35. 虽然称死是这样，总是不受任何限制。
# 36. 新浪网络的新闻站点，假装是报道中国。
# 37. 去世界纪念� some practical tips to make it work for you. Ready? Let's dive in!

# Understanding The Child Tax Credit

# The child tax credit is like a financial incentive that helps offset some of the costs associated with raising a child. It's calculated based on your income, the number of children you have, and the number of classes is 6. We need to find out how many students are in each class.

# Participant A: Right, so we need to divide the total number of students by the number of classes, correct?

# Participant B: Exactly. We can use integer division here