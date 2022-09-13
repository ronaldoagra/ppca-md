
    import pandas as pd
    from pycaret.MLUsecase.CLASSIFICATION import load_model, predict_model
    from fastapi import FastAPI
    import uvicorn
    # Create the app
    app = FastAPI()
    # Load trained Pipeline
    model = load_model('api/bow_multi_api')
    # Define predict function
    @app.post('/predict')
    def predict(abdomen, abracar, abraco, acesso, acho, acionar, acometer, acometidos, acompanhamento, acontece, acontecera, acordo, acougue, acrescido, acrescimo, acrescimos, aderi, aderir, aderiram, aderiu, adesao, adesoes, adianta, adiantou, adicionais, adicional, adocao, adota, adotadas, adotando, adotar, adotaram, adotarem, adotou, aferir, afetadas, agencia, agir, agora, agua, ainda, ajuda, ajudara, albergaria, albergues, alcool, alguem, algum, alguma, alimentar, alojam, alojar, ambiente, ambientes, animais, antigo, anvisa, aparece, apareceu, aparelho, aperto, aplicabilidade, apoio, apos, apresenta, aprovacao, aptos, ar, arca, area, arma, ase, assim, assintomatico, associou, asua, ate, atende, atender, atendimento, atinge, ativacao, ativado, atividade, atividades, ativo, atrativos, atraves, atua, atualizacao, atualizacoes, atualizado, atualizar, atualizarei, atuando, aumenta, aumentam, aumentar, aumentarao, aumento, auto, autorizacao, avalia, bacilo, bacteria, bacteriana, bacterias, baixa, baixar, bancos, banheiro, base, bebedouro, beber, bebida, beijo, beijos, beneficios, bexiga, boas, boca, br, brasil, brasileiro, bubcar, busco, cabelo, cada, cadastrado, cadastrados, cadastramento, cadastrar, cadastro, cadastrotur, cadastrur, cadastur, cadtrur, cadtur, caminhada, cancelar, capacitados, carne, carro, casa, caseiro, caso, causa, causam, cd, cedula, central, cerebral, cerebro, certeza, cgu, chama, chance, chegar, cheio, cheiro, china, chineses, cidadao, circulacao, clientes, cliquei, cobertura, cobrada, cobrado, cobrar, cobrir, code, coemcou, coisa, coisas, colocando, comecou, comer, comerciais, comercio, comida, commerce, comorbidade, compartilhada, compartilhar, compensar, completar, comprar, compras, comprometidos, comprova, comprovacao, comprovocao, comum, comuns, conceito, condicao, condicionado, condicoes, condominio, confiar, confiavel, confiro, conformidade, conjunto, conseguindo, conseguir, consideracao, consigo, construidos, consultar, consulto, consumo, conta, contagiado, contagiar, contagio, contaminacao, contaminada, contaminadas, contaminado, contaminados, contaminar, contato, conter, continua, continuar, contra, contrair, contraprova, contribuir, contribuira, controladoria, controle, conversa, conversei, coronavirus, corpo, correios, correm, corrente, corro, corticoides, cov, covid, covide, credenciados, criacao, criada, crianca, criar, criara, criterios, crivo, cuidado, cuidados, cumpre, cumprimento, cumprindo, cumprira, cura, curado, curativos, custa, custo, custos, dada, dara, debaixo, declaracao, defina, definicao, deixa, deixar, delivery, demanda, demora, dentro, denuncia, denunciar, denuncio, desativado, descobrir, descreva, descumprimento, descumprimentos, desenvolver, desfiliar, desisti, desligar, despesas, desses, deste, destina, destinado, destino, determinado, deu, deve, devem, devera, devido, devo, diagnosticada, diagnostico, diaria, diarias, dias, diferencial, diga, dinheiro, direito, disperta, disponiveis, disseminadas, distancia, divulgadas, divulgar, divulgaram, diz, dizer, documentacao, documento, doenc, doenca, doencas, doente, dois, dor, doses, dowload, download, durante, duvida, efetivar, eficazes, elevador, embalagens, embratur, embutir, emissao, emitir, empreedimento, empreendedores, empreendiemtos, empreendimento, empreendimentos, emprego, emprendimento, empresa, empresas, encaixa, encomenda, encomendas, engole, enquadrado, ensina, ensine, entidades, entrada, entrar, entrega, entregas, erro, esclareca, espaco, espalha, especificas, especifico, especificos, espera, esperar, espirra, espirre, espirro, estabeleceu, estabelecidas, estabelecido, estabelecidos, estabelecimento, estabelecimentos, estabelecimentto, estado, estalagem, estando, estao, estrangeiro, estrangeiros, evitar, exame, exames, excluir, exemplo, exigencias, exigida, exigidas, exigidos, exigir, exigiu, exiigir, existe, existem, explica, explique, exportar, exterior, externo, extraordinarias, extras, fabricar, facil, faco, fala, falar, fale, faliu, familia, fara, farei, fatores, favorecido, fazem, fazendo, fazer, fechado, fechados, fechei, feito, feridas, fezes, fica, ficalizara, ficar, ficarei, filho, filiado, finalidade, fiscalizacao, fiscalizacoes, fiscalizadas, fiscalizado, fiscalizados, fiscalizar, fizeram, flotel, forma, formas, formulacao, forte, frequente, frequentes, frutas, fumantes, funciona, funcionar, funcionario, ganglinar, ganglionar, ganglios, ganha, ganhando, ganharei, ganho, garante, garantir, garantira, garganta, gasta, gasto, gel, gerais, geral, geram, gerar, gerencial, globo, gostaria, gosto, goticulas, governo, graca, gratuito, grave, graves, gripado, gripe, ha, habilitados, havera, hevera, higiene, higienizacao, higienizar, higiente, hiv, hospedagem, hospedaria, hospedei, hostalagem, hoteis, hotel, hotelaria, hoteleiro, hoteleito, hotes, humanos, identificar, impacto, impactos, impede, impedido, impedir, implementacao, implicara, importante, importar, impostas, imposto, impostos, imprimir, incentivando, incentivo, incluri, indeterminado, infeccao, infecccao, infecciona, infectada, infectadas, infectados, infectam, infestacao, informacao, informacoes, informado, inscrever, inserir, instestinal, instrucao, integrada, intenstinal, interessado, interesse, internacionais, internet, interno, intestinal, intestino, intestinos, ira, irregularidade, irregularidades, ja, koch, laboratorio, laboratorios, lavar, lei, leite, lesoes, leva, levantada, levar, levou, liberacao, liberar, limpar, limpeza, linfatico, linfaticos, locais, local, localidades, login, lotericas, lugar, luz, maior, mal, manter, mao, mascara, massificacao, mata, materiais, material, medidas, meio, meios, melhor, melhores, meningea, mercado, miliar, minima, miningite, ministerio, moedas, monitorar, morcego, mostra, mostrara, moteis, motiva, motivados, motivo, mtur, muda, multa, multado, mundo, nacional, nada, nao, natureza, necessarias, necessario, necessarios, negocio, negocios, nervoso, neurologica, neurotuberculose, nivel, normas, notificada, novamente, novas, novo, numero, obeso, objeto, objetos, obriga, obrigado, obrigando, obrigatoriedade, obrigatorio, observados, observei, obter, obtive, ocorre, ocular, oculos, oftalmologica, olho, olhos, oms, onde, onibus, opas, opcao, opcional, operante, option, orgaos, orientacao, orientacoes, orientadas, orientados, oriundo, oriundos, ossatura, ossea, osso, ossos, outra, outras, ouvidoria, ouvidos, oximetria, paciente, pacote, padaria, paga, pagamento, pagar, pagarei, pago, painel, pais, paises, pandemia, pangolim, papel, par, parente, participar, participo, passa, passar, passarao, pecas, peco, pedido, pedir, pega, pegar, pego, pele, pensoes, pequisar, perceber, perigosa, periodicamente, periodicidade, permanece, permite, perspectiva, perto, pescoco, pesquiso, pessoa, pessoas, plasticas, plataforma, pleitar, pleitear, pleura, pleural, pode, podem, podera, poderao, poeira, porque, portar, posicionar, possa, possivel, posso, possuira, possuo, postais, pousada, pousadas, pq, pra, pracas, praticados, praticar, praticas, prazo, pre, precaver, precisa, precisam, preciso, precos, preencher, presente, prestador, prestadores, prestados, pretende, prevencao, prevenir, previne, previsao, previstas, principais, proceder, procedimento, processar, processo, procuro, produtos, produzidos, produzir, profissionais, programa, promocionais, promovera, propagacao, propostas, propostos, proprietario, proprio, proprios, protecao, proteger, protegido, protejo, protocolo, protocolos, provoca, provocar, publicitario, publico, pulmao, pulmoes, pulmonar, punicao, punicoes, qr, qto, quais, qualquer, quantas, quanto, quantos, quer, queria, quero, quimica, quiser, ramo, ratear, realizar, receber, recebi, receita, recolher, recomendacoes, recomendadas, recomendado, recomendados, recomensacoes, rede, reducao, reduzir, reduzira, referente, regem, registrado, registrados, registrar, registro, regra, regras, relacao, relacionadas, remedio, remunerar, reponsavel, representa, representara, representatividade, requerer, requisito, requisitos, resorts, respira, respiratorias, responsavel, retirar, retomada, reveste, rins, risco, riscos, rosto, roupa, roupas, sabendo, saber, sacola, sacolas, sair, sairam, saiu, sanguinea, sanitaria, sanitarias, sao, sarar, sars, saude, secrecoes, sedex, segmento, segue, seguido, seguindo, seguir, segura, seguranca, seguras, seguro, seguros, sei, sel, selecionei, selo, selos, sempre, sendo, sentido, sentindo, sentira, sentirao, sera, serao, serve, servico, servicos, setor, setores, setorias, severa, sido, significa, sim, similares, sintoma, sintomas, sistema, site, situacao, so, sobre, sobrevive, sofrerao, sol, solicitacao, solicitar, status, submetidos, sucesso, sujos, superficies, supermercado, surgir, surjam, sus, suspendi, suspensa, suspensao, suspensas, tabaco, tambem, tapete, taxa, taxas, tbsnc, tecido, tecidos, tecnico, tempo, ter, tera, terao, terbeculose, teste, tiberculose, tipo, tipos, tira, tirar, toalha, tocando, tocar, todo, todos, tomar, torax, tornar, tornara, tosse, tossir, trabalho, transmissao, transmite, transmitida, transmitido, transmitidos, transmitir, transportadoras, transversais, trara, trasmitida, tratamento, tratar, traves, treinamento, tres, troca, tuberculose, tuberculoso, turismo, turista, turistas, turistica, turisticas, turistico, turisticos, tussa, uber, umida, unhas, uniao, unico, urinaria, urinario, usa, usar, uso, usuario, usuarios, usufluir, utensilhos, utilizados, utilizando, utilizar, vacina, vacinar, vacinas, vai, validacao, validade, validados, valor, vantagem, vantagens, vantajoso, vao, variacao, varias, varios, varrer, vc, veio, vejo, vem, vendi, ver, verduras, verificar, verificara, verifico, via, viagem, viajar, vias, vigilancia, vindo, vir, virilha, virus, visitar, visivel, vistoria, vistorias, vivo, voltar, vou):
        data = pd.DataFrame([[abdomen, abracar, abraco, acesso, acho, acionar, acometer, acometidos, acompanhamento, acontece, acontecera, acordo, acougue, acrescido, acrescimo, acrescimos, aderi, aderir, aderiram, aderiu, adesao, adesoes, adianta, adiantou, adicionais, adicional, adocao, adota, adotadas, adotando, adotar, adotaram, adotarem, adotou, aferir, afetadas, agencia, agir, agora, agua, ainda, ajuda, ajudara, albergaria, albergues, alcool, alguem, algum, alguma, alimentar, alojam, alojar, ambiente, ambientes, animais, antigo, anvisa, aparece, apareceu, aparelho, aperto, aplicabilidade, apoio, apos, apresenta, aprovacao, aptos, ar, arca, area, arma, ase, assim, assintomatico, associou, asua, ate, atende, atender, atendimento, atinge, ativacao, ativado, atividade, atividades, ativo, atrativos, atraves, atua, atualizacao, atualizacoes, atualizado, atualizar, atualizarei, atuando, aumenta, aumentam, aumentar, aumentarao, aumento, auto, autorizacao, avalia, bacilo, bacteria, bacteriana, bacterias, baixa, baixar, bancos, banheiro, base, bebedouro, beber, bebida, beijo, beijos, beneficios, bexiga, boas, boca, br, brasil, brasileiro, bubcar, busco, cabelo, cada, cadastrado, cadastrados, cadastramento, cadastrar, cadastro, cadastrotur, cadastrur, cadastur, cadtrur, cadtur, caminhada, cancelar, capacitados, carne, carro, casa, caseiro, caso, causa, causam, cd, cedula, central, cerebral, cerebro, certeza, cgu, chama, chance, chegar, cheio, cheiro, china, chineses, cidadao, circulacao, clientes, cliquei, cobertura, cobrada, cobrado, cobrar, cobrir, code, coemcou, coisa, coisas, colocando, comecou, comer, comerciais, comercio, comida, commerce, comorbidade, compartilhada, compartilhar, compensar, completar, comprar, compras, comprometidos, comprova, comprovacao, comprovocao, comum, comuns, conceito, condicao, condicionado, condicoes, condominio, confiar, confiavel, confiro, conformidade, conjunto, conseguindo, conseguir, consideracao, consigo, construidos, consultar, consulto, consumo, conta, contagiado, contagiar, contagio, contaminacao, contaminada, contaminadas, contaminado, contaminados, contaminar, contato, conter, continua, continuar, contra, contrair, contraprova, contribuir, contribuira, controladoria, controle, conversa, conversei, coronavirus, corpo, correios, correm, corrente, corro, corticoides, cov, covid, covide, credenciados, criacao, criada, crianca, criar, criara, criterios, crivo, cuidado, cuidados, cumpre, cumprimento, cumprindo, cumprira, cura, curado, curativos, custa, custo, custos, dada, dara, debaixo, declaracao, defina, definicao, deixa, deixar, delivery, demanda, demora, dentro, denuncia, denunciar, denuncio, desativado, descobrir, descreva, descumprimento, descumprimentos, desenvolver, desfiliar, desisti, desligar, despesas, desses, deste, destina, destinado, destino, determinado, deu, deve, devem, devera, devido, devo, diagnosticada, diagnostico, diaria, diarias, dias, diferencial, diga, dinheiro, direito, disperta, disponiveis, disseminadas, distancia, divulgadas, divulgar, divulgaram, diz, dizer, documentacao, documento, doenc, doenca, doencas, doente, dois, dor, doses, dowload, download, durante, duvida, efetivar, eficazes, elevador, embalagens, embratur, embutir, emissao, emitir, empreedimento, empreendedores, empreendiemtos, empreendimento, empreendimentos, emprego, emprendimento, empresa, empresas, encaixa, encomenda, encomendas, engole, enquadrado, ensina, ensine, entidades, entrada, entrar, entrega, entregas, erro, esclareca, espaco, espalha, especificas, especifico, especificos, espera, esperar, espirra, espirre, espirro, estabeleceu, estabelecidas, estabelecido, estabelecidos, estabelecimento, estabelecimentos, estabelecimentto, estado, estalagem, estando, estao, estrangeiro, estrangeiros, evitar, exame, exames, excluir, exemplo, exigencias, exigida, exigidas, exigidos, exigir, exigiu, exiigir, existe, existem, explica, explique, exportar, exterior, externo, extraordinarias, extras, fabricar, facil, faco, fala, falar, fale, faliu, familia, fara, farei, fatores, favorecido, fazem, fazendo, fazer, fechado, fechados, fechei, feito, feridas, fezes, fica, ficalizara, ficar, ficarei, filho, filiado, finalidade, fiscalizacao, fiscalizacoes, fiscalizadas, fiscalizado, fiscalizados, fiscalizar, fizeram, flotel, forma, formas, formulacao, forte, frequente, frequentes, frutas, fumantes, funciona, funcionar, funcionario, ganglinar, ganglionar, ganglios, ganha, ganhando, ganharei, ganho, garante, garantir, garantira, garganta, gasta, gasto, gel, gerais, geral, geram, gerar, gerencial, globo, gostaria, gosto, goticulas, governo, graca, gratuito, grave, graves, gripado, gripe, ha, habilitados, havera, hevera, higiene, higienizacao, higienizar, higiente, hiv, hospedagem, hospedaria, hospedei, hostalagem, hoteis, hotel, hotelaria, hoteleiro, hoteleito, hotes, humanos, identificar, impacto, impactos, impede, impedido, impedir, implementacao, implicara, importante, importar, impostas, imposto, impostos, imprimir, incentivando, incentivo, incluri, indeterminado, infeccao, infecccao, infecciona, infectada, infectadas, infectados, infectam, infestacao, informacao, informacoes, informado, inscrever, inserir, instestinal, instrucao, integrada, intenstinal, interessado, interesse, internacionais, internet, interno, intestinal, intestino, intestinos, ira, irregularidade, irregularidades, ja, koch, laboratorio, laboratorios, lavar, lei, leite, lesoes, leva, levantada, levar, levou, liberacao, liberar, limpar, limpeza, linfatico, linfaticos, locais, local, localidades, login, lotericas, lugar, luz, maior, mal, manter, mao, mascara, massificacao, mata, materiais, material, medidas, meio, meios, melhor, melhores, meningea, mercado, miliar, minima, miningite, ministerio, moedas, monitorar, morcego, mostra, mostrara, moteis, motiva, motivados, motivo, mtur, muda, multa, multado, mundo, nacional, nada, nao, natureza, necessarias, necessario, necessarios, negocio, negocios, nervoso, neurologica, neurotuberculose, nivel, normas, notificada, novamente, novas, novo, numero, obeso, objeto, objetos, obriga, obrigado, obrigando, obrigatoriedade, obrigatorio, observados, observei, obter, obtive, ocorre, ocular, oculos, oftalmologica, olho, olhos, oms, onde, onibus, opas, opcao, opcional, operante, option, orgaos, orientacao, orientacoes, orientadas, orientados, oriundo, oriundos, ossatura, ossea, osso, ossos, outra, outras, ouvidoria, ouvidos, oximetria, paciente, pacote, padaria, paga, pagamento, pagar, pagarei, pago, painel, pais, paises, pandemia, pangolim, papel, par, parente, participar, participo, passa, passar, passarao, pecas, peco, pedido, pedir, pega, pegar, pego, pele, pensoes, pequisar, perceber, perigosa, periodicamente, periodicidade, permanece, permite, perspectiva, perto, pescoco, pesquiso, pessoa, pessoas, plasticas, plataforma, pleitar, pleitear, pleura, pleural, pode, podem, podera, poderao, poeira, porque, portar, posicionar, possa, possivel, posso, possuira, possuo, postais, pousada, pousadas, pq, pra, pracas, praticados, praticar, praticas, prazo, pre, precaver, precisa, precisam, preciso, precos, preencher, presente, prestador, prestadores, prestados, pretende, prevencao, prevenir, previne, previsao, previstas, principais, proceder, procedimento, processar, processo, procuro, produtos, produzidos, produzir, profissionais, programa, promocionais, promovera, propagacao, propostas, propostos, proprietario, proprio, proprios, protecao, proteger, protegido, protejo, protocolo, protocolos, provoca, provocar, publicitario, publico, pulmao, pulmoes, pulmonar, punicao, punicoes, qr, qto, quais, qualquer, quantas, quanto, quantos, quer, queria, quero, quimica, quiser, ramo, ratear, realizar, receber, recebi, receita, recolher, recomendacoes, recomendadas, recomendado, recomendados, recomensacoes, rede, reducao, reduzir, reduzira, referente, regem, registrado, registrados, registrar, registro, regra, regras, relacao, relacionadas, remedio, remunerar, reponsavel, representa, representara, representatividade, requerer, requisito, requisitos, resorts, respira, respiratorias, responsavel, retirar, retomada, reveste, rins, risco, riscos, rosto, roupa, roupas, sabendo, saber, sacola, sacolas, sair, sairam, saiu, sanguinea, sanitaria, sanitarias, sao, sarar, sars, saude, secrecoes, sedex, segmento, segue, seguido, seguindo, seguir, segura, seguranca, seguras, seguro, seguros, sei, sel, selecionei, selo, selos, sempre, sendo, sentido, sentindo, sentira, sentirao, sera, serao, serve, servico, servicos, setor, setores, setorias, severa, sido, significa, sim, similares, sintoma, sintomas, sistema, site, situacao, so, sobre, sobrevive, sofrerao, sol, solicitacao, solicitar, status, submetidos, sucesso, sujos, superficies, supermercado, surgir, surjam, sus, suspendi, suspensa, suspensao, suspensas, tabaco, tambem, tapete, taxa, taxas, tbsnc, tecido, tecidos, tecnico, tempo, ter, tera, terao, terbeculose, teste, tiberculose, tipo, tipos, tira, tirar, toalha, tocando, tocar, todo, todos, tomar, torax, tornar, tornara, tosse, tossir, trabalho, transmissao, transmite, transmitida, transmitido, transmitidos, transmitir, transportadoras, transversais, trara, trasmitida, tratamento, tratar, traves, treinamento, tres, troca, tuberculose, tuberculoso, turismo, turista, turistas, turistica, turisticas, turistico, turisticos, tussa, uber, umida, unhas, uniao, unico, urinaria, urinario, usa, usar, uso, usuario, usuarios, usufluir, utensilhos, utilizados, utilizando, utilizar, vacina, vacinar, vacinas, vai, validacao, validade, validados, valor, vantagem, vantagens, vantajoso, vao, variacao, varias, varios, varrer, vc, veio, vejo, vem, vendi, ver, verduras, verificar, verificara, verifico, via, viagem, viajar, vias, vigilancia, vindo, vir, virilha, virus, visitar, visivel, vistoria, vistorias, vivo, voltar, vou]])
        data.columns = ['abdomen', 'abracar', 'abraco', 'acesso', 'acho', 'acionar', 'acometer', 'acometidos', 'acompanhamento', 'acontece', 'acontecera', 'acordo', 'acougue', 'acrescido', 'acrescimo', 'acrescimos', 'aderi', 'aderir', 'aderiram', 'aderiu', 'adesao', 'adesoes', 'adianta', 'adiantou', 'adicionais', 'adicional', 'adocao', 'adota', 'adotadas', 'adotando', 'adotar', 'adotaram', 'adotarem', 'adotou', 'aferir', 'afetadas', 'agencia', 'agir', 'agora', 'agua', 'ainda', 'ajuda', 'ajudara', 'albergaria', 'albergues', 'alcool', 'alguem', 'algum', 'alguma', 'alimentar', 'alojam', 'alojar', 'ambiente', 'ambientes', 'animais', 'antigo', 'anvisa', 'aparece', 'apareceu', 'aparelho', 'aperto', 'aplicabilidade', 'apoio', 'apos', 'apresenta', 'aprovacao', 'aptos', 'ar', 'arca', 'area', 'arma', 'ase', 'assim', 'assintomatico', 'associou', 'asua', 'ate', 'atende', 'atender', 'atendimento', 'atinge', 'ativacao', 'ativado', 'atividade', 'atividades', 'ativo', 'atrativos', 'atraves', 'atua', 'atualizacao', 'atualizacoes', 'atualizado', 'atualizar', 'atualizarei', 'atuando', 'aumenta', 'aumentam', 'aumentar', 'aumentarao', 'aumento', 'auto', 'autorizacao', 'avalia', 'bacilo', 'bacteria', 'bacteriana', 'bacterias', 'baixa', 'baixar', 'bancos', 'banheiro', 'base', 'bebedouro', 'beber', 'bebida', 'beijo', 'beijos', 'beneficios', 'bexiga', 'boas', 'boca', 'br', 'brasil', 'brasileiro', 'bubcar', 'busco', 'cabelo', 'cada', 'cadastrado', 'cadastrados', 'cadastramento', 'cadastrar', 'cadastro', 'cadastrotur', 'cadastrur', 'cadastur', 'cadtrur', 'cadtur', 'caminhada', 'cancelar', 'capacitados', 'carne', 'carro', 'casa', 'caseiro', 'caso', 'causa', 'causam', 'cd', 'cedula', 'central', 'cerebral', 'cerebro', 'certeza', 'cgu', 'chama', 'chance', 'chegar', 'cheio', 'cheiro', 'china', 'chineses', 'cidadao', 'circulacao', 'clientes', 'cliquei', 'cobertura', 'cobrada', 'cobrado', 'cobrar', 'cobrir', 'code', 'coemcou', 'coisa', 'coisas', 'colocando', 'comecou', 'comer', 'comerciais', 'comercio', 'comida', 'commerce', 'comorbidade', 'compartilhada', 'compartilhar', 'compensar', 'completar', 'comprar', 'compras', 'comprometidos', 'comprova', 'comprovacao', 'comprovocao', 'comum', 'comuns', 'conceito', 'condicao', 'condicionado', 'condicoes', 'condominio', 'confiar', 'confiavel', 'confiro', 'conformidade', 'conjunto', 'conseguindo', 'conseguir', 'consideracao', 'consigo', 'construidos', 'consultar', 'consulto', 'consumo', 'conta', 'contagiado', 'contagiar', 'contagio', 'contaminacao', 'contaminada', 'contaminadas', 'contaminado', 'contaminados', 'contaminar', 'contato', 'conter', 'continua', 'continuar', 'contra', 'contrair', 'contraprova', 'contribuir', 'contribuira', 'controladoria', 'controle', 'conversa', 'conversei', 'coronavirus', 'corpo', 'correios', 'correm', 'corrente', 'corro', 'corticoides', 'cov', 'covid', 'covide', 'credenciados', 'criacao', 'criada', 'crianca', 'criar', 'criara', 'criterios', 'crivo', 'cuidado', 'cuidados', 'cumpre', 'cumprimento', 'cumprindo', 'cumprira', 'cura', 'curado', 'curativos', 'custa', 'custo', 'custos', 'dada', 'dara', 'debaixo', 'declaracao', 'defina', 'definicao', 'deixa', 'deixar', 'delivery', 'demanda', 'demora', 'dentro', 'denuncia', 'denunciar', 'denuncio', 'desativado', 'descobrir', 'descreva', 'descumprimento', 'descumprimentos', 'desenvolver', 'desfiliar', 'desisti', 'desligar', 'despesas', 'desses', 'deste', 'destina', 'destinado', 'destino', 'determinado', 'deu', 'deve', 'devem', 'devera', 'devido', 'devo', 'diagnosticada', 'diagnostico', 'diaria', 'diarias', 'dias', 'diferencial', 'diga', 'dinheiro', 'direito', 'disperta', 'disponiveis', 'disseminadas', 'distancia', 'divulgadas', 'divulgar', 'divulgaram', 'diz', 'dizer', 'documentacao', 'documento', 'doenc', 'doenca', 'doencas', 'doente', 'dois', 'dor', 'doses', 'dowload', 'download', 'durante', 'duvida', 'efetivar', 'eficazes', 'elevador', 'embalagens', 'embratur', 'embutir', 'emissao', 'emitir', 'empreedimento', 'empreendedores', 'empreendiemtos', 'empreendimento', 'empreendimentos', 'emprego', 'emprendimento', 'empresa', 'empresas', 'encaixa', 'encomenda', 'encomendas', 'engole', 'enquadrado', 'ensina', 'ensine', 'entidades', 'entrada', 'entrar', 'entrega', 'entregas', 'erro', 'esclareca', 'espaco', 'espalha', 'especificas', 'especifico', 'especificos', 'espera', 'esperar', 'espirra', 'espirre', 'espirro', 'estabeleceu', 'estabelecidas', 'estabelecido', 'estabelecidos', 'estabelecimento', 'estabelecimentos', 'estabelecimentto', 'estado', 'estalagem', 'estando', 'estao', 'estrangeiro', 'estrangeiros', 'evitar', 'exame', 'exames', 'excluir', 'exemplo', 'exigencias', 'exigida', 'exigidas', 'exigidos', 'exigir', 'exigiu', 'exiigir', 'existe', 'existem', 'explica', 'explique', 'exportar', 'exterior', 'externo', 'extraordinarias', 'extras', 'fabricar', 'facil', 'faco', 'fala', 'falar', 'fale', 'faliu', 'familia', 'fara', 'farei', 'fatores', 'favorecido', 'fazem', 'fazendo', 'fazer', 'fechado', 'fechados', 'fechei', 'feito', 'feridas', 'fezes', 'fica', 'ficalizara', 'ficar', 'ficarei', 'filho', 'filiado', 'finalidade', 'fiscalizacao', 'fiscalizacoes', 'fiscalizadas', 'fiscalizado', 'fiscalizados', 'fiscalizar', 'fizeram', 'flotel', 'forma', 'formas', 'formulacao', 'forte', 'frequente', 'frequentes', 'frutas', 'fumantes', 'funciona', 'funcionar', 'funcionario', 'ganglinar', 'ganglionar', 'ganglios', 'ganha', 'ganhando', 'ganharei', 'ganho', 'garante', 'garantir', 'garantira', 'garganta', 'gasta', 'gasto', 'gel', 'gerais', 'geral', 'geram', 'gerar', 'gerencial', 'globo', 'gostaria', 'gosto', 'goticulas', 'governo', 'graca', 'gratuito', 'grave', 'graves', 'gripado', 'gripe', 'ha', 'habilitados', 'havera', 'hevera', 'higiene', 'higienizacao', 'higienizar', 'higiente', 'hiv', 'hospedagem', 'hospedaria', 'hospedei', 'hostalagem', 'hoteis', 'hotel', 'hotelaria', 'hoteleiro', 'hoteleito', 'hotes', 'humanos', 'identificar', 'impacto', 'impactos', 'impede', 'impedido', 'impedir', 'implementacao', 'implicara', 'importante', 'importar', 'impostas', 'imposto', 'impostos', 'imprimir', 'incentivando', 'incentivo', 'incluri', 'indeterminado', 'infeccao', 'infecccao', 'infecciona', 'infectada', 'infectadas', 'infectados', 'infectam', 'infestacao', 'informacao', 'informacoes', 'informado', 'inscrever', 'inserir', 'instestinal', 'instrucao', 'integrada', 'intenstinal', 'interessado', 'interesse', 'internacionais', 'internet', 'interno', 'intestinal', 'intestino', 'intestinos', 'ira', 'irregularidade', 'irregularidades', 'ja', 'koch', 'laboratorio', 'laboratorios', 'lavar', 'lei', 'leite', 'lesoes', 'leva', 'levantada', 'levar', 'levou', 'liberacao', 'liberar', 'limpar', 'limpeza', 'linfatico', 'linfaticos', 'locais', 'local', 'localidades', 'login', 'lotericas', 'lugar', 'luz', 'maior', 'mal', 'manter', 'mao', 'mascara', 'massificacao', 'mata', 'materiais', 'material', 'medidas', 'meio', 'meios', 'melhor', 'melhores', 'meningea', 'mercado', 'miliar', 'minima', 'miningite', 'ministerio', 'moedas', 'monitorar', 'morcego', 'mostra', 'mostrara', 'moteis', 'motiva', 'motivados', 'motivo', 'mtur', 'muda', 'multa', 'multado', 'mundo', 'nacional', 'nada', 'nao', 'natureza', 'necessarias', 'necessario', 'necessarios', 'negocio', 'negocios', 'nervoso', 'neurologica', 'neurotuberculose', 'nivel', 'normas', 'notificada', 'novamente', 'novas', 'novo', 'numero', 'obeso', 'objeto', 'objetos', 'obriga', 'obrigado', 'obrigando', 'obrigatoriedade', 'obrigatorio', 'observados', 'observei', 'obter', 'obtive', 'ocorre', 'ocular', 'oculos', 'oftalmologica', 'olho', 'olhos', 'oms', 'onde', 'onibus', 'opas', 'opcao', 'opcional', 'operante', 'option', 'orgaos', 'orientacao', 'orientacoes', 'orientadas', 'orientados', 'oriundo', 'oriundos', 'ossatura', 'ossea', 'osso', 'ossos', 'outra', 'outras', 'ouvidoria', 'ouvidos', 'oximetria', 'paciente', 'pacote', 'padaria', 'paga', 'pagamento', 'pagar', 'pagarei', 'pago', 'painel', 'pais', 'paises', 'pandemia', 'pangolim', 'papel', 'par', 'parente', 'participar', 'participo', 'passa', 'passar', 'passarao', 'pecas', 'peco', 'pedido', 'pedir', 'pega', 'pegar', 'pego', 'pele', 'pensoes', 'pequisar', 'perceber', 'perigosa', 'periodicamente', 'periodicidade', 'permanece', 'permite', 'perspectiva', 'perto', 'pescoco', 'pesquiso', 'pessoa', 'pessoas', 'plasticas', 'plataforma', 'pleitar', 'pleitear', 'pleura', 'pleural', 'pode', 'podem', 'podera', 'poderao', 'poeira', 'porque', 'portar', 'posicionar', 'possa', 'possivel', 'posso', 'possuira', 'possuo', 'postais', 'pousada', 'pousadas', 'pq', 'pra', 'pracas', 'praticados', 'praticar', 'praticas', 'prazo', 'pre', 'precaver', 'precisa', 'precisam', 'preciso', 'precos', 'preencher', 'presente', 'prestador', 'prestadores', 'prestados', 'pretende', 'prevencao', 'prevenir', 'previne', 'previsao', 'previstas', 'principais', 'proceder', 'procedimento', 'processar', 'processo', 'procuro', 'produtos', 'produzidos', 'produzir', 'profissionais', 'programa', 'promocionais', 'promovera', 'propagacao', 'propostas', 'propostos', 'proprietario', 'proprio', 'proprios', 'protecao', 'proteger', 'protegido', 'protejo', 'protocolo', 'protocolos', 'provoca', 'provocar', 'publicitario', 'publico', 'pulmao', 'pulmoes', 'pulmonar', 'punicao', 'punicoes', 'qr', 'qto', 'quais', 'qualquer', 'quantas', 'quanto', 'quantos', 'quer', 'queria', 'quero', 'quimica', 'quiser', 'ramo', 'ratear', 'realizar', 'receber', 'recebi', 'receita', 'recolher', 'recomendacoes', 'recomendadas', 'recomendado', 'recomendados', 'recomensacoes', 'rede', 'reducao', 'reduzir', 'reduzira', 'referente', 'regem', 'registrado', 'registrados', 'registrar', 'registro', 'regra', 'regras', 'relacao', 'relacionadas', 'remedio', 'remunerar', 'reponsavel', 'representa', 'representara', 'representatividade', 'requerer', 'requisito', 'requisitos', 'resorts', 'respira', 'respiratorias', 'responsavel', 'retirar', 'retomada', 'reveste', 'rins', 'risco', 'riscos', 'rosto', 'roupa', 'roupas', 'sabendo', 'saber', 'sacola', 'sacolas', 'sair', 'sairam', 'saiu', 'sanguinea', 'sanitaria', 'sanitarias', 'sao', 'sarar', 'sars', 'saude', 'secrecoes', 'sedex', 'segmento', 'segue', 'seguido', 'seguindo', 'seguir', 'segura', 'seguranca', 'seguras', 'seguro', 'seguros', 'sei', 'sel', 'selecionei', 'selo', 'selos', 'sempre', 'sendo', 'sentido', 'sentindo', 'sentira', 'sentirao', 'sera', 'serao', 'serve', 'servico', 'servicos', 'setor', 'setores', 'setorias', 'severa', 'sido', 'significa', 'sim', 'similares', 'sintoma', 'sintomas', 'sistema', 'site', 'situacao', 'so', 'sobre', 'sobrevive', 'sofrerao', 'sol', 'solicitacao', 'solicitar', 'status', 'submetidos', 'sucesso', 'sujos', 'superficies', 'supermercado', 'surgir', 'surjam', 'sus', 'suspendi', 'suspensa', 'suspensao', 'suspensas', 'tabaco', 'tambem', 'tapete', 'taxa', 'taxas', 'tbsnc', 'tecido', 'tecidos', 'tecnico', 'tempo', 'ter', 'tera', 'terao', 'terbeculose', 'teste', 'tiberculose', 'tipo', 'tipos', 'tira', 'tirar', 'toalha', 'tocando', 'tocar', 'todo', 'todos', 'tomar', 'torax', 'tornar', 'tornara', 'tosse', 'tossir', 'trabalho', 'transmissao', 'transmite', 'transmitida', 'transmitido', 'transmitidos', 'transmitir', 'transportadoras', 'transversais', 'trara', 'trasmitida', 'tratamento', 'tratar', 'traves', 'treinamento', 'tres', 'troca', 'tuberculose', 'tuberculoso', 'turismo', 'turista', 'turistas', 'turistica', 'turisticas', 'turistico', 'turisticos', 'tussa', 'uber', 'umida', 'unhas', 'uniao', 'unico', 'urinaria', 'urinario', 'usa', 'usar', 'uso', 'usuario', 'usuarios', 'usufluir', 'utensilhos', 'utilizados', 'utilizando', 'utilizar', 'vacina', 'vacinar', 'vacinas', 'vai', 'validacao', 'validade', 'validados', 'valor', 'vantagem', 'vantagens', 'vantajoso', 'vao', 'variacao', 'varias', 'varios', 'varrer', 'vc', 'veio', 'vejo', 'vem', 'vendi', 'ver', 'verduras', 'verificar', 'verificara', 'verifico', 'via', 'viagem', 'viajar', 'vias', 'vigilancia', 'vindo', 'vir', 'virilha', 'virus', 'visitar', 'visivel', 'vistoria', 'vistorias', 'vivo', 'voltar', 'vou']
        predictions = predict_model(model, data=data)
        return {'prediction': list(predictions['Label'])}
    if __name__ == '__main__':
        uvicorn.run(app, host='127.0.0.1', port=8000)